import functools
from typing import Tuple

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tf_ops import explained_variance
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.tune import register_trainable

torch, nn = try_import_torch()
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"
OTHER_AGENT = "other_agent"

from abc import ABC, abstractmethod

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from custom_model import *

torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module, ABC):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # env parameters
        self.obs_space_shape = obs_space.shape[0]
        self.act_space_shape = action_space.n
        self.centralized = model_config["custom_options"]["critic"]["centralized"]
        self.max_num_agents = model_config["custom_options"]["max_num_agents"]
        self.max_num_opponents = self.max_num_agents - 1
        self.debug_mode = True

        # Build the actor network
        self.actor = self._build_actor(**model_config["custom_options"]["actor"])
        self.register_variables(self.actor.variables)

        # Central Value Network
        self.central_critic = self._build_critic(**model_config["custom_options"]["critic"])
        self.register_variables(self.critic.variables)

    @abstractmethod
    def _build_actor(self, **kwargs) -> torch.nn.Module:
        pass

    @abstractmethod
    def _build_critic(self, **kwargs) -> torch.nn.Module:
        pass

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        action = self.actor(input_dict["obs_flat"])
        return action, state

    def central_value_function(self, obs, opponent_obs, opponent_actions):
        input_ = torch.cat([
            obs, opponent_obs,
            torch.nn.functional.one_hot(opponent_actions, 2).float()
        ], 1)
        return torch.reshape(self.central_critic(input_), [-1])

    # def central_value_function(self, obs, other_agent):
    #     if self.centralized:
    #         return tf.reshape(self.critic([obs, other_agent]), [-1])
    #     return tf.reshape(self.critic(obs), [-1])

    @override(ModelV2)
    def value_function(self):
        return self.model.value_function()  # not used


class CCPPO_Concatenation(TorchCentralizedCriticModel):

    def _build_actor(self, hidden_layers=[512, 512, 512], **kwargs):
        actor = Actor(num_inputs=self.obs_space_shape,
                      hidden_layers=hidden_layers,
                      num_outputs=self.act_space_shape)
        return actor

    def _build_critic(self, hidden_layers=[512, 512, 512], **kwargs, ):
        num_inputs = (self.obs_space_shape + self.act_space_shape) * self.max_num_agents
        critic = CentralizedCritic(num_inputs=num_inputs,
                                   hidden_layers=hidden_layers,
                                   num_outputs=1)
        return critic


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the code."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.

def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"

    # one hot encoding parser
    one_hot_enc = functools.partial(one_hot_encoding, n_classes=policy.action_space.n)
    max_num_opponents = policy.model.max_num_opponents

    if policy.loss_initialized():
        assert other_agent_batches is not None

        if len(other_agent_batches) > max_num_opponents:
            raise ValueError(
                "The number of opponents is too large, got {} (max at {})".format(
                    len(other_agent_batches), max_num_opponents
                )
            )

        # lifespan of the agents
        time_span = (sample_batch["t"][0], sample_batch["t"][-1])

        # agents whose time overlaps with the current agent time_span
        # returns agent_id: agent_time_span, opp_sample_batch
        opponents = [
            Opponent(
                (opp_batch["t"][0], opp_batch["t"][-1]),
                opp_batch[SampleBatch.CUR_OBS],
                one_hot_enc(opp_batch[SampleBatch.ACTIONS]),
            )
            for agent_id, (_, opp_batch) in other_agent_batches.items()
            if time_overlap(time_span, (opp_batch["t"][0], opp_batch["t"][-1]))
        ]

        # apply the adequate cropping or padding compared to time_span
        for opp in opponents:
            opp.crop_or_pad(time_span)

        # add a padding for the missing opponents
        missing_opponent = Opponent(
            None,
            np.zeros_like(sample_batch[SampleBatch.CUR_OBS]),
            one_hot_enc(np.zeros_like(sample_batch[SampleBatch.ACTIONS])),
        )
        opponents = opponents + (
                [missing_opponent] * (max_num_opponents - len(opponents))
        )

        # add random permutation of the opponents
        perm = np.random.permutation(np.arange(max_num_opponents))
        opponents = [opponents[p] for p in perm]

        # add the oppponents' information into sample_batch
        sample_batch[OTHER_AGENT] = np.concatenate(
            [opp.observation for opp in opponents] + [opp.action for opp in opponents],
            axis=-1,
        )
        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_value_function(
            sample_batch[SampleBatch.CUR_OBS], sample_batch[OTHER_AGENT]
        )

    else:

        # opponents' observation placeholder
        missing_obs = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        missing_act = one_hot_enc(np.zeros_like(sample_batch[SampleBatch.ACTIONS]))
        sample_batch[OTHER_AGENT] = np.concatenate(
            [missing_obs for _ in range(max_num_opponents)]
            + [missing_act for _ in range(max_num_opponents)],
            axis=-1,
        )

        # value prediction
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS], dtype=np.float32
        )

    train_batch = compute_advantages(
        sample_batch,
        0.0,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OPPONENT_OBS],
        train_batch[OPPONENT_ACTION])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out),
    }


class Opponent(object):
    def __init__(
            self, time_span: Tuple[int, int], observation: np.ndarray, action: np.ndarray
    ):
        self.time_span = time_span
        self.observation = observation
        self.action = action

    def crop_or_pad(self, reference_time_span):
        time_difference = self._get_time_difference(reference_time_span)
        for key in self.__dict__:
            if key == "time_span":
                continue
            setattr(
                self, key, Opponent._crop_or_pad(getattr(self, key), *time_difference)
            )

    def _get_time_difference(self, reference):
        lower = reference[0] - self.time_span[0]
        upper = self.time_span[1] - reference[1]
        return lower, upper

    @staticmethod
    def _crop_or_pad(values, lower, upper):
        values = values[max(lower, 0):]
        values = values[: len(values) - max(upper, 0)]
        values = np.pad(
            values,
            pad_width=[
                (-min(lower, 0), -min(0, upper)),
                *[(0, 0) for k in range(values.ndim - 1)],
            ],
            mode="constant",
        )
        return values


def time_overlap(time_span, agent_time):
    """Check if agent_time overlaps with time_span"""
    return agent_time[0] <= time_span[1] and agent_time[1] >= time_span[0]


def one_hot_encoding(values, n_classes):
    return np.eye(n_classes)[values]


CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    # grad_stats_fn=central_vf_stats,
    mixins=[
        TorchLR,
        TorchEntropyCoeffSchedule,
        TorchKLCoeffMixin,
        CentralizedValueMixin
    ])

register_trainable(
    "CCPPO_Concatenation",
    PPOTrainer.with_updates(
        name="CCPPOTrainer", get_policy_class=lambda c: CCPPOTorchPolicy
    ),
)

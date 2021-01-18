import functools
import os
from typing import Tuple

import ray
from ray import tune
import argparse
from gym.spaces import Discrete

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import \
    CentralizedCriticModel, TorchCentralizedCriticModel
from ray.rllib.models import ModelCatalog
from ray.rllib.models.jax.misc import np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.tune import register_trainable

torch, nn = try_import_torch()
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"

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

        opponents = [
            Opponent(
                (opp_batch["t"][0], opp_batch["t"][-1]),
                opp_batch[SampleBatch.CUR_OBS],
                one_hot_enc(opp_batch[SampleBatch.ACTIONS]),
            )
            for agent_id, (_, opp_batch) in other_agent_batches.items()
            if time_overlap(time_span, (opp_batch["t"][0], opp_batch["t"][-1]))
        ]




        [(_, opponent_batch)] = list(other_agent_batches.values())

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
        sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

        # overwrite default VF prediction with the central VF

        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[SampleBatch.CUR_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_OBS], policy.device),
            convert_to_torch_tensor(
                sample_batch[OPPONENT_ACTION], policy.device)) \
            .cpu().detach().numpy()

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[OPPONENT_OBS] = np.zeros_like(
            sample_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.zeros_like(
            sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
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


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


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



def one_hot_encoding(values, n_classes):
    return np.eye(n_classes)[values]



CCPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="CCPPOTorchPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_init=setup_torch_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        TorchLR,
        TorchEntropyCoeffSchedule,
        TorchKLCoeffMixin,
        CentralizedValueMixin
    ])

register_trainable(
    "CcTransformer",
    PPOTrainer.with_updates(
        name="CCPPOTrainer", get_policy_class=lambda c: CCPPOTorchPolicy
    ),
)


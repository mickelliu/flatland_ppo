import functools

from abc import ABC, abstractmethod
import numpy as np
import os
from typing import Tuple
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, PPOLoss, PPOTFPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils.explained_variance import explained_variance
from ray.tune import register_trainable

OTHER_AGENT = "other_agent"


class CentralizedCriticModel(ABC, TFModelV2):
    """Multi-agent model that implements a centralized VF."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )  # The Method Resolution Order (MRO) will manage the dependencies.

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
        self.critic = self._build_critic(**model_config["custom_options"]["critic"])
        self.register_variables(self.critic.variables)

        # summaries
        if self.debug_mode:
            print("Actor Model:\n", self.actor.summary())
            print("Critic Model:\n", self.critic.summary())

    @abstractmethod
    def _build_actor(self, **kwargs) -> tf.keras.Model:
        pass

    @abstractmethod
    def _build_critic(self, **kwargs) -> tf.keras.Model:
        pass

    def forward(self, input_dict, state, seq_lens):
        policy = self.actor(input_dict["obs_flat"])
        self._value_out = tf.reduce_mean(input_tensor=policy, axis=-1)  # not used
        return policy, state

    def central_value_function(self, obs, other_agent):
        if self.centralized:
            return tf.reshape(self.critic([obs, other_agent]), [-1])
        return tf.reshape(self.critic(obs), [-1])

    def value_function(self):
        return tf.reshape(self._value_out, [1])  # not used


def build_fullyConnected(
    inputs, hidden_layers, num_outputs, activation_fn="relu", name=None
):
    name = name or "fc_network"  # default_name

    # Fully connected hidden layers
    x = inputs
    for k, layer_size in enumerate(hidden_layers):
        x = tf.keras.layers.Dense(
            layer_size,
            name="{}/fc_{}".format(name, k),
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer=tf.keras.initializers.constant(0.1),
        )(x)

    # output layer
    output = tf.keras.layers.Dense(
        num_outputs,
        name="{}/fc_out".format(name),
        activation=None,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        bias_initializer=tf.keras.initializers.constant(0.1),
    )(x)

    return output


class CcConcatenate(CentralizedCriticModel):
    """Multi-agent model that implements a centralized VF."""

    def _build_actor(
        self, activation_fn="relu", hidden_layers=[512, 512, 512], **kwargs
    ):

        inputs = tf.keras.layers.Input(shape=(self.obs_space_shape,), name="obs")

        output = build_fullyConnected(
            inputs=inputs,
            hidden_layers=hidden_layers,
            num_outputs=self.act_space_shape,
            activation_fn=activation_fn,
            name="actor",
        )

        return tf.keras.Model(inputs, output)

    def _build_critic(
        self,
        activation_fn="relu",
        hidden_layers=[512, 512, 512],
        centralized=True,
        **kwargs,
    ):
        obs = tf.keras.layers.Input(shape=(self.obs_space_shape,), name="obs")
        inputs = [obs]

        if centralized:
            other_agent = tf.keras.layers.Input(
                shape=(
                    (self.obs_space_shape + self.act_space_shape)
                    * self.max_num_opponents,
                ),
                name="other_agent",
            )
            inputs += [other_agent]
            input_layer = tf.keras.layers.Concatenate(axis=1)(inputs)
        else:
            input_layer = obs

        output = build_fullyConnected(
            inputs=input_layer,
            hidden_layers=hidden_layers,
            num_outputs=1,
            activation_fn=activation_fn,
            name="critic",
        )

        return tf.keras.Model(inputs, output)


class CentralizedValueMixin(object):
    """Add methods to evaluate the central value function from the model."""

    # the sample batch need to be put in a placeholder before
    # being feed to the network, otherwise it will redefine the tensor dimensions
    def __init__(self):
        self.central_value_function = self.model.central_value_function(
            self.get_placeholder(SampleBatch.CUR_OBS), self.get_placeholder(OTHER_AGENT)
        )

    def compute_central_value_function(
        self, obs, other_agent
    ):  # opponent_obs, opponent_actions):
        feed_dict = {
            self.get_placeholder(SampleBatch.CUR_OBS): obs,
            self.get_placeholder(OTHER_AGENT): other_agent,
        }
        return self.get_session().run(self.central_value_function, feed_dict)


# Grabs the other obs/policy and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches=None, episode=None
):
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


# Copied from PPO but optimizing the central value function
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)

    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    policy.central_value_out = policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS], train_batch[OTHER_AGENT]
    )

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        policy.central_value_out,
        policy.kl_coeff,
        tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool),
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def setup_mixins(policy, obs_space, action_space, config):
    # copied from PPO
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.central_value_out
        )
    }


def one_hot_encoding(values, n_classes):
    return np.eye(n_classes)[values]


def time_overlap(time_span, agent_time):
    """Check if agent_time overlaps with time_span"""
    return agent_time[0] <= time_span[1] and agent_time[1] >= time_span[0]


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
        values = values[max(lower, 0) :]
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


CCPPOPolicy = PPOTFPolicy.with_updates(
    name="CCPPOPolicy",
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=loss_with_central_critic,
    before_loss_init=setup_mixins,
    grad_stats_fn=central_vf_stats,
    mixins=[
        LearningRateSchedule,
        EntropyCoeffSchedule,
        KLCoeffMixin,
        CentralizedValueMixin,
    ],
)
register_trainable(
    "CcConcatenate",
    PPOTrainer.with_updates(
        name="CCPPOTrainer", get_policy_class=lambda c: CCPPOPolicy
    ),
)

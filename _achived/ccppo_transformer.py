import functools
from abc import ABC, abstractmethod
from typing import Tuple

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, \
    KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models import ModelV2
from ray.rllib.models.jax.misc import np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tf_ops import explained_variance
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.tune import register_trainable

torch, nn = try_import_torch()
OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


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


class CCPPO_Transformer(TorchCentralizedCriticModel):

    def _build_actor(self, hidden_layers=[512, 512, 512], **kwargs):
        actor = Actor(num_inputs=self.obs_space_shape,
                      hidden_layers=hidden_layers,
                      num_outputs=self.act_space_shape)
        return actor

    def _build_critic(
            self,
            hidden_layers=[512, 512, 512],
            centralized=True,
            num_layers=1,
            embedding_size=128,
            num_heads=8,
            d_model=256,
            **kwargs,
    ):

        critic = CentralizedCritic(
                                   hidden_layers=hidden_layers,
                                   num_outputs=1,
                                   tfr_config={
                                            "num_layers": num_layers,
                                            "dim_model": d_model,
                                            "num_heads": num_heads,
                                            "dim_feedforward": embedding_size
                                        }
                                   )

        # agent's input
        agent_hidden_layers = [2 * embedding_size, embedding_size]
        agent_dims = [self.obs_space_shape, *agent_hidden_layers, self.embedding_size]
        agent_embedding = build_fullyConnected(layers_dims=agent_dims)

        # opponents' input
        opponent_shape = [self.max_num_opponents, (self.obs_space_shape + self.act_space_shape)]
        opponent_hidden_layers = [2 * embedding_size, embedding_size]
        opponent_dims = [opponent_shape[0], *opponent_hidden_layers, self.embedding_size]
        opponent_embedding = build_fullyConnected(layers_dims=opponent_dims)

        # opponents' embedding
        # `[batch_size, self.max_num_opponents, embedding_size]`

        # # output shape: `[batch_size, 1, d_model]`
        # opponents_embedding = MultiHeadAttentionLayer(
        #     num_heads=num_heads, d_model=d_model, use_scale=use_scale
        # )(
        #     [queries, opponent_embedding, opponent_embedding]
        # )  # `[q, k, v]`

        # multi-head attention
        queries = torch.unsqueeze(agent_embedding, dim=1)  # number of queries = 1

        multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        attn_output, _ = multihead_attn(queries, opponent_embedding, opponent_embedding)

        # remove the addtional dimension
        opponents_embedding = torch.squeeze(attn_output, axis=1)

        # `[batch_size, embedding_size + d_model]`
        embeddings = torch.cat([agent_embedding, opponents_embedding], axis=-1)

        output = build_fullyConnected(
            layers_dims=[embeddings.shape(1), hidden_layers, 1]
        )  # `[batch_size, ]`

        return


def build_fullyConnected(layers_dims, input_layer=None):
    # Fully connected hidden layers
    if not input_layer:
        model = [input_layer]
    else:
        model = []

    for k in range(len(layers_dims) - 1):

        model.append(nn.Linaer(layers_dims[k], layers_dims[k + 1]))
        if k < len(layers_dims) - 1:
            model.append(nn.ReLU())

    return nn.Sequential(*model)


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

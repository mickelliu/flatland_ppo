# from abc import ABC, abstractmethod
#
# from gym.spaces import Box
#
# from ray.rllib.models.modelv2 import ModelV2
# from ray.rllib.models.torch.misc import SlimFC
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
# from ray.rllib.utils.annotations import override
# from ray.rllib.utils.framework import try_import_tf, try_import_torch
#
# torch, nn = try_import_torch()
#
#
# class TorchCentralizedCriticModel(TorchModelV2, nn.Module, ABC):
#     """Multi-agent model that implements a centralized VF."""
#
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)
#
#         # env parameters
#         self.obs_space_shape = obs_space.shape[0]
#         self.act_space_shape = action_space.n
#         self.centralized = model_config["custom_options"]["critic"]["centralized"]
#         self.max_num_agents = model_config["custom_options"]["max_num_agents"]
#         self.max_num_opponents = self.max_num_agents - 1
#         self.debug_mode = True
#
#         # Build the actor network
#         self.actor = self._build_actor(**model_config["custom_options"]["actor"])
#         self.register_variables(self.actor.variables)
#
#         # Central Value Network
#         self.central_critic = self._build_critic(**model_config["custom_options"]["critic"])
#         self.register_variables(self.critic.variables)
#
#     @abstractmethod
#     def _build_actor(self, **kwargs) -> torch.nn.Module:
#         pass
#
#     @abstractmethod
#     def _build_critic(self, **kwargs) -> torch.nn.Module:
#         pass
#
#     @override(ModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         action = self.actor(input_dict["obs_flat"])
#         return action, state
#
#     def central_value_function(self, obs, opponent_obs, opponent_actions):
#         input_ = torch.cat([
#             obs, opponent_obs,
#             torch.nn.functional.one_hot(opponent_actions, 2).float()
#         ], 1)
#         return torch.reshape(self.central_critic(input_), [-1])
#
#     # def central_value_function(self, obs, other_agent):
#     #     if self.centralized:
#     #         return tf.reshape(self.critic([obs, other_agent]), [-1])
#     #     return tf.reshape(self.critic(obs), [-1])
#
#     @override(ModelV2)
#     def value_function(self):
#         return self.model.value_function()  # not used
#
#
# class CCPPO_Transformer(TorchCentralizedCriticModel):
#
#     def _build_actor(self, hidden_layers=[512, 512, 512], **kwargs):
#         layers_dims = [self.obs_space_shape, *hidden_layers, self.act_space_shape]
#
#         return build_fullyConnected(layers_dims=layers_dims)
#
#     def _build_critic(
#             self,
#             hidden_layers=[512, 512, 512],
#             centralized=True,
#             embedding_size=128,
#             num_heads=8,
#             d_model=256,
#             use_scale=True,
#             **kwargs,
#     ):
#         # agent's input
#         agent_hidden_layers = [2 * embedding_size, embedding_size]
#         agent_dims = [self.obs_space_shape, *agent_hidden_layers, self.embedding_size]
#         agent_embedding = build_fullyConnected(layers_dims=agent_dims)
#
#         # opponents' input
#         opponent_shape = [self.max_num_opponents, (self.obs_space_shape + self.act_space_shape)]
#         opponent_hidden_layers = [2 * embedding_size, embedding_size]
#         opponent_dims = [opponent_shape[0], *opponent_hidden_layers, self.embedding_size]
#         opponent_embedding = build_fullyConnected(layers_dims=opponent_dims)
#
#         # opponents' embedding
#         # `[batch_size, self.max_num_opponents, embedding_size]`
#
#         # # output shape: `[batch_size, 1, d_model]`
#         # opponents_embedding = MultiHeadAttentionLayer(
#         #     num_heads=num_heads, d_model=d_model, use_scale=use_scale
#         # )(
#         #     [queries, opponent_embedding, opponent_embedding]
#         # )  # `[q, k, v]`
#
#         # multi-head attention
#         queries = torch.unsqueeze(agent_embedding, dim=1)  # number of queries = 1
#
#         multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
#         attn_output, _ = multihead_attn(queries, opponent_embedding, opponent_embedding)
#
#         # remove the addtional dimension
#         opponents_embedding = torch.squeeze(attn_output, axis=1)
#
#         # `[batch_size, embedding_size + d_model]`
#         embeddings = torch.cat([agent_embedding, opponents_embedding], axis=-1)
#
#         output = build_fullyConnected(
#             layers_dims=[embeddings.shape(1), hidden_layers, 1]
#         )  # `[batch_size, ]`
#
#         return
#
#
# def build_fullyConnected(layers_dims, input_layer=None):
#     # Fully connected hidden layers
#     if not input_layer:
#         model = [input_layer]
#     else:
#         model = []
#
#     for k in range(len(layers_dims) - 1):
#
#         model.append(nn.Linaer(layers_dims[k], layers_dims[k + 1]))
#         if k < len(layers_dims) - 1:
#             model.append(nn.ReLU())
#
#     return nn.Sequential(*model)
#
#
# class Opponent(object):
#     def __init__(
#             self, time_span: Tuple[int, int], observation: np.ndarray, action: np.ndarray
#     ):
#         self.time_span = time_span
#         self.observation = observation
#         self.action = action
#
#     def crop_or_pad(self, reference_time_span):
#         time_difference = self._get_time_difference(reference_time_span)
#         for key in self.__dict__:
#             if key == "time_span":
#                 continue
#             setattr(
#                 self, key, Opponent._crop_or_pad(getattr(self, key), *time_difference)
#             )
#
#     def _get_time_difference(self, reference):
#         lower = reference[0] - self.time_span[0]
#         upper = self.time_span[1] - reference[1]
#         return lower, upper
#
#     @staticmethod
#     def _crop_or_pad(values, lower, upper):
#         values = values[max(lower, 0):]
#         values = values[: len(values) - max(upper, 0)]
#         values = np.pad(
#             values,
#             pad_width=[
#                 (-min(lower, 0), -min(0, upper)),
#                 *[(0, 0) for k in range(values.ndim - 1)],
#             ],
#             mode="constant",
#         )
#         return values

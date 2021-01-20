import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


# ===================== actor =====================
class Actor(nn.Module):

    def __init__(self, num_inputs, hidden_layers, num_outputs):
        super(Actor, self).__init__()
        self.shape = num_inputs
        self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.out = nn.Linear(hidden_layers[2], num_outputs)

    def forward(self, x):
        x = x.view(-1, self.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


#===================== critic =====================
class CentralizedCritic(nn.Module):

    def __init__(self, num_inputs, hidden_layers, num_outputs):
        super(CentralizedCritic, self).__init__()

        self.shape = num_inputs
        self.fc1 = nn.Linear(num_inputs, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.out = nn.Linear(hidden_layers[2], num_outputs)

    def forward(self, x):
        x = x.view(-1, self.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


class CentralizedCriticWithTransformer(nn.Module):

    def __init__(self, d_model, hidden_layers, num_outputs, tfr_config):
        super(CentralizedCritic, self).__init__()

        self.shape = d_model
        self.transformer = TransformerEncoder(
            num_layers=tfr_config["num_layers"],
            dim_model=tfr_config["dim_model"],
            num_heads=tfr_config["num_heads"],
            dim_feedforward=tfr_config["dim_feedforward"]
        )

        self.fc1 = nn.Linear(d_model, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.out = nn.Linear(hidden_layers[2], num_outputs)

    def forward(self, x):
        x = x.view(-1, self.shape)
        x = self.transformer(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(d_model, d_model, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    # def get_attention_maps(self, x, mask=None):
    #     attention_maps = []
    #     for l in self.layers:
    #         _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
    #         attention_maps.append(attn_map)
    #         x = l(x)
    #     return attention_maps




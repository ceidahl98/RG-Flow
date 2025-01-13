from math import sqrt
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .scale import Scale
from .swish import Swish





class ResNet(nn.Module):
    def __init__(self, num_res_blocks, widths, final_scale, final_tanh, use_attention=True, num_heads=4):
        assert widths[0] == widths[-1]

        super().__init__()
        self.use_attention = use_attention

        self.res_blocks = nn.ModuleList(
            [self.build_res_block(widths) for _ in range(num_res_blocks)]
        )

        if self.use_attention:
            self.attention_layers = nn.ModuleList(
                [self.build_attention(widths[0], num_heads) for _ in range(num_res_blocks)]
            )

        if final_scale:
            self.scale = Scale(widths[-1])
        else:
            self.scale = None

        if final_tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        # 2.81 is the gain for swish
        bound = sqrt(2.81 * 3 / in_features)
        init.uniform_(linear.weight, -bound, bound)
        init.zeros_(linear.bias)
        linear = weight_norm(linear)
        return linear

    def build_res_block(self, widths):
        layers = []
        for i in range(len(widths) - 1):
            layers.append(Swish(widths[i]))
            layers.append(self.build_linear(widths[i], widths[i + 1]))
        return nn.Sequential(*layers)

    def build_attention(self, embed_dim, num_heads):
        # Multihead attention layer
        return nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        for i, res_block in enumerate(self.res_blocks):
            # Residual block
            res_out = res_block(x)
            x = (x + res_out) / sqrt(2)

            # Attention layer (if enabled)
            if self.use_attention:
                # Add a sequence dimension for attention
                x_seq = x.unsqueeze(1)  # Add sequence dimension
                x, _ = self.attention_layers[i](x_seq, x_seq, x_seq)
                x = x.squeeze(1)  # Remove sequence dimension

        if self.scale:
            x = self.scale(x)
        if self.tanh:
            x = self.tanh(x)

        return x

class ResNetReshape(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        shape = x.shape    # (B*num_RG_blocks, C, K, K)
        x = x.view(shape[0], -1)    # (B*num_RG_blocks, C*K*K)
        x = super().forward(x)
        x = x.view(shape)
        return x

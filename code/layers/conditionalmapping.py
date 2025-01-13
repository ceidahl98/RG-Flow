from math import sqrt
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import weight_norm

from .scale import Scale
from .swish import Swish


class GPTConfig:
    block_size: int = 64
    vocab_size: int = 64
    n_layer: int = 10
    n_head: int = 4
    n_embd: int = 64
    dropout: float=0.1
    action_space = 2


class MLP(nn.Module):

    def __init__(self,n_embd,dropout):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd,4*n_embd)
        self.activation = nn.SiLU()
        self.d1 = nn.Dropout(dropout)
        self.c_proj = nn.Linear(4*n_embd,n_embd)
        self.c_proj.INIT_SCALE = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.d1(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd,n_head,dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd,dropout)

    def forward(self,x,context):
        q = self.ln_1(context)
        k = self.ln_1(x)
        v = self.ln_1(x)

        x = x + self.attn(q,k,v)[0][:,-1,:].unsqueeze(1)
        x = x + self.mlp(self.ln_2(x))
        return x



class ConditionalMapping(nn.Module):
    def __init__(self, patch_size,channels, embd_dim,n_head, num_layers,dropout):
        super().__init__()
        input_size = int(patch_size**2*channels)
        output_size = int(input_size*3)
        self.block = Block(embd_dim,n_head,dropout)
        self.num_layers = num_layers
        self.input_embedding = nn.Linear(input_size,embd_dim)
        self.context_embedding = nn.Linear(int(input_size/4),embd_dim)
        self.proj = nn.Linear(embd_dim,output_size)

    def forward(self,x,context):
        device = next(self.parameters()).device
        x = x.to(device)
        context = context.to(device)
        b,c,num_blocks,p_area = context.shape
        _,_,x_num_blocks,x_p_area = x.shape

        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        x = x.view(int(b*x_num_blocks),-1,int(x_p_area*c))
        context = context.permute(0,2,1,3).flatten(start_dim=2)
        context = context.view(int(b*x_num_blocks),-1,int(p_area*c))

        x = self.input_embedding(x)
        context = self.context_embedding(context)
        for layer in range(self.num_layers):
            x = self.block(x,context)

        x = self.proj(x)
        x = nn.functional.relu(x)
        x = x.view(b,num_blocks,-1)
        #print(x.shape,"AHAHAHA")
        return x

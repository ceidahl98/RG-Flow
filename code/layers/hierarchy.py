import torch
from torch import nn

import utils

from .flow import Flow
#rom layers.resnet import ConditionalMapping
from layers.conditionalmapping import ConditionalMapping
#add poisson dirichlet process and mappings

class HierarchyBijector(Flow):
    def __init__(self, indexI, indexJ, layers, prior=None):
        super().__init__(prior)
        assert len(layers) == len(indexI)
        assert len(layers) == len(indexJ)

        patch_size=4
        channels=3
        embd_dim=512
        n_head = 4
        num_layers = 4
        dropout=.1
        self.rate=10
        self.layers = nn.ModuleList(layers)
        self.indexI = indexI
        self.indexJ = indexJ
        self.latents = []
        self.conditional_mappings = []
        for i in range(len(layers)):
            self.conditional_mappings.append(
                ConditionalMapping(patch_size,channels,embd_dim,n_head,num_layers,dropout)
            )


    def forward(self, x):
        # dim(x) = (B, C, H, W)
        batch_size = x.shape[0]

        ldj = x.new_zeros(batch_size)
        latents = []
        contexts = []

        depth=1


        for layer, indexI, indexJ in zip(self.layers, self.indexI,
                                         self.indexJ):



            x, x_ = utils.dispatch(indexI, indexJ, x)

            if depth % 2 == 1 and depth!=1:
                b, c, num_blocks, p = x_.shape
                contexts.append(x_.clone().detach().reshape(b, c, -1, p // 4))

            x_ = utils.stackRGblock(x_)

            x_, log_prob = layer.forward(x_)
            x_ = nn.functional.relu(x_)
            x_=torch.clamp(x_,max=self.rate*10)
            ldj = ldj + log_prob.view(batch_size, -1).sum(dim=1)

            x_ = utils.unstackRGblock(x_, batch_size)

            if depth %2 ==0: #collect latents at each level
                latents.append(x_.clone().detach())

            x = utils.collect(indexI, indexJ, x, x_)

            depth+=1

        in_latents = latents[1:]
        out_latents = latents[:-1]


        out_mappings = []
        for con_map,in_latent,context in zip(self.conditional_mappings,in_latents,contexts):
            mapping = torch.clamp(con_map(in_latent,context),max=self.rate*10)
            out_mappings.append(
                mapping
            )

        con_loss = torch.tensor(0.0,device=x.device)
        for out,out_mapping in zip(out_latents,out_mappings):
            out = utils.stackRGblock(out)
            mask = torch.ones_like(out, dtype=torch.bool, device=x.device)
            mask[:, :, 1::2, ::2] = False
            out = out[mask].view(batch_size, 3, -1, 12)
            out = out.permute(0, 2, 1, 3).flatten(start_dim=2)
            print(out_mapping.max(),"OUTMAX")
            con_loss += nn.functional.mse_loss(out_mapping.to(x.device),out.to(x.device))


        return x, ldj, con_loss

    def inverse(self, z):
        batch_size = z.shape[0]

        inv_ldj = z.new_zeros(batch_size)
        for layer, indexI, indexJ in reversed(
                list(zip(self.layers, self.indexI, self.indexJ))):
            z, z_ = utils.dispatch(indexI, indexJ, z)
            print(z_.shape,"Z_")
            z_ = utils.stackRGblock(z_)

            z_, log_prob = layer.inverse(z_)
            inv_ldj = inv_ldj + log_prob.view(batch_size, -1).sum(dim=1)

            z_ = utils.unstackRGblock(z_, batch_size)
            z = utils.collect(indexI, indexJ, z, z_)

        return z, inv_ldj

from torch import nn
import torch

from .source import Source
class Poisson(Source):
    def __init__(self, nvars, scale=1):
        super().__init__(nvars)
        self.nvars = nvars
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.get_default_dtype()))

        self.register_buffer(

            'rate', torch.tensor(5,dtype=torch.get_default_dtype())

        )


    def sample(self, batch_size):
        n_samples = 1000

        shape = [batch_size]+self.nvars
        rate = self.rate.expand(shape)

        z = torch.poisson(rate)
        # exp = torch.distributions.exponential.Exponential(rate)
        #
        # dt = exp.rsample((n_samples,))
        #
        # t = torch.cumsum(dt, dim=0)
        # ind = nn.functional.sigmoid((1 - t) / 1)

        # z = torch.sum(ind, dim=0)

        return z

    def log_prob(self, count):

        rate = self.rate
        count = nn.functional.relu(count)
        count = torch.clamp(count,max=10*rate)
        print(count.min(),"count_min",count.max(),"count_max")
        log_prob = count * torch.log(rate) - rate - torch.lgamma(count + 1)
        log_prob = torch.sum(log_prob.view(log_prob.shape[0],-1),dim=1)
        return log_prob

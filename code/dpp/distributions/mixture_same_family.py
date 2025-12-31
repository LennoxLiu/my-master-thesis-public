import torch
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from scipy.optimize import brentq
import torch.distributions as D
from torch.distributions.normal import Normal


class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, x):
        x = self._pad(x)
        
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        sf_x = 1 - self.component_distribution.cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(torch.log(sf_x) + mix_logits, dim=-1)
    

    


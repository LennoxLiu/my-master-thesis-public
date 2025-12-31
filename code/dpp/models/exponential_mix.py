import torch
import torch.nn as nn
import torch.distributions as D

from dpp.distributions import TransformedDistribution
from dpp.utils import clamp_preserve_gradients
from .recurrent_tpp import RecurrentTPP

class ExponentialMixtureDistribution(TransformedDistribution):
    """
    Mixture of exponential distributions for normalized log-inter-event times.

    We model the distribution as:
        u ~ Mixture of Exponentials (rates, weights)
        t = exp(u * std_log_inter_time + mean_log_inter_time)

    Args:
        log_rates: Logarithms of rate parameters for component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Mean of log inter-event times (for normalization)
        std_log_inter_time: Std of log inter-event times (for normalization)
        num_samples: Number of samples for Monte Carlo entropy estimation
    """
    def __init__(
        self,
        log_rates: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        num_samples: int = 1_000,
    ):
        # Ensure valid distributions
        rates = clamp_preserve_gradients(log_rates, -3, 3).exp()
        weights = torch.softmax(log_weights, dim=-1)
        
        # Create mixture distribution
        mixture_dist = D.Categorical(probs=weights)
        component_dist = D.Exponential(rates)
        base_dist = D.MixtureSameFamily(mixture_dist, component_dist)
        
        # Define transformations
        transforms = [
            D.ExpTransform().inv,  # t -> log(t)
            D.AffineTransform(
                loc=-mean_log_inter_time, 
                scale=1.0 / std_log_inter_time
            )  # log(t) -> (log(t) - mean)/std
        ]
        
        super().__init__(base_dist, transforms)
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.log_rates = log_rates
        self.log_weights = log_weights
        self.num_samples = num_samples

    @property
    def mean(self) -> torch.Tensor:
        """Compute expected value of normalized log-inter-event times."""
        # Get mixture weights and rates
        weights = torch.softmax(self.log_weights, dim=-1)
        rates = clamp_preserve_gradients(self.log_rates, -3, 3).exp()
        
        # E[log(t)] for exponential: -log(λ) - γ (Euler's constant)
        e_log_t = -torch.log(rates) - torch.tensor(0.57721566490153286)
        
        # Weighted average and normalization
        weighted_e_log_t = (weights * e_log_t).sum(dim=-1)
        return (weighted_e_log_t - self.mean_log_inter_time) / self.std_log_inter_time

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy using Monte Carlo sampling.
        
        The entropy is computed as:
            H(Y) = H(X) - log(std) - E[log(X)]
        where:
            X ~ Mixture of Exponentials
            Y = (log(X) - mean) / std
        """
        # Sample from base distribution (inter-event times)
        x = self.base_dist.sample((self.num_samples,))  # (num_samples, *batch_shape)
        
        # Compute log probability of samples
        log_p_x = self.base_dist.log_prob(x)  # (num_samples, *batch_shape)
        
        # Compute entropy components
        H_X = -log_p_x.mean(0)  # H(X) estimate
        E_log_x = torch.log(x).mean(0)  # E[log(X)]
        
        # Adjustment terms
        std_tensor = torch.tensor(
            self.std_log_inter_time,
            dtype=x.dtype,
            device=x.device
        )
        adjustment = torch.log(std_tensor) + E_log_x
        
        return H_X - adjustment

class ExponentialMix(RecurrentTPP):
    """
    RNN-based TPP model using a mixture of exponentials for inter-event times.

    Args:
        num_marks: Number of distinct event types
        mean_log_inter_time: Mean of log inter-event times (normalization)
        std_log_inter_time: Std of log inter-event times (normalization)
        context_size: Dimension of context vector
        mark_embedding_size: Dimension of mark embeddings
        num_mix_components: Number of mixture components
        hidden_sizes: List of hidden layer sizes for MLP
        max_seq_len: Maximum sequence length
        num_samples: Number of samples for Monte Carlo entropy estimation
    """
    def __init__(
        self,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        num_mix_components: int = 16,
        hidden_sizes: list = [16],
        max_seq_len: int = 10,
        num_samples: int = 1_000,
    ):
        super().__init__(
            num_marks=num_marks,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            context_size=context_size,
            mark_embedding_size=mark_embedding_size,
            max_seq_len=max_seq_len,
        )
        self.num_mix_components = num_mix_components
        self.num_samples = num_samples
        
        # Build MLP to map context to parameters
        layers = []
        input_size = self.context_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.25))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 2 * self.num_mix_components))
        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.mlp.apply(init_weights)

    def get_inter_time_dist(self, context: torch.Tensor) -> D.Distribution:
        """
        Get exponential mixture distribution given context.
        
        Args:
            context: History embedding, shape (batch_size, context_size)
            
        Returns:
            dist: Mixture distribution for normalized log-inter-event times
        """
        # Get raw parameters from MLP
        raw_params = self.mlp(context)
        
        # Split into rate and weight parameters
        log_rates = raw_params[..., :self.num_mix_components]
        log_weights = raw_params[..., self.num_mix_components:]
        
        # Clamp rates for stability
        log_rates = clamp_preserve_gradients(log_rates, -3, 3)
        
        return ExponentialMixtureDistribution(
            log_rates=log_rates,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
            num_samples=self.num_samples
        )
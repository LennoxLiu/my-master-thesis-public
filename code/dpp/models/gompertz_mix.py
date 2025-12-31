import torch
import torch.nn as nn
import torch.distributions as D

from dpp.distributions import TransformedDistribution
from dpp.utils import clamp_preserve_gradients
from .recurrent_tpp import RecurrentTPP
from torch.types import _size

class GompertzDistribution(D.Distribution):
    """
    Gompertz distribution with parameters alpha (scale) and beta (shape).
    
    PDF: p(τ|α,β) = α exp(βτ - (α/β) exp(βτ) + α/β)
    
    Args:
        alphas: Scale parameters, shape (batch_size, seq_len, num_mix_components)
        betas: Shape parameters, shape (batch_size, seq_len, num_mix_components)
    """
    def __init__(self, alphas: torch.Tensor, betas: torch.Tensor, validate_args=False):
        self.alphas = alphas
        self.betas = betas
        batch_shape = alphas.shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of the distribution at given value.
        
        Args:
            value: Inter-event times, shape (batch_size, 1)
            
        Returns:
            log_prob: Log probability, shape (batch_size, )
        """
        # Ensure value can be broadcasted to the shape of alphas/betas
        # If value is (batch_size, 1), unsqueeze to (batch_size, 1, 1) for broadcasting
        if value.dim() == 2 and self.alphas.dim() == 3:
            value = value.unsqueeze(-1) # (batch_size, 1, 1)
            
        # print(f"Gompertz log_prob input: {value.shape}, alphas: {self.alphas.shape}, betas: {self.betas.shape}")
        # Compute log probability using Gompertz formula
        self.betas = self.betas.clamp(min=1e-10)  # Avoid division by zero
        self.alphas = self.alphas.clamp(min=1e-10)  # Avoid log(0)
        term1 = torch.log(self.alphas)  # Avoid log(0)
        term2 = self.betas * value
        term2 = term2.clamp(max=5)   # Avoid overflow in exp
        term3 = (self.alphas / self.betas) * (torch.exp(term2) - 1)
        log_prob = term1 + term2 - term3 + (self.alphas / self.betas)  # Avoid division by zero
        
        # if term1.isinf().any():
        #     raise ValueError("term1: Log probability contains infinite values. Check input parameters.")
        # if term2.isinf().any():
        #     raise ValueError("term2: Log probability contains inf values. Check input parameters.")
        # if term3.isinf().any():
        #     raise ValueError("term3: Log probability contains inf values. Check input parameters.")
        # if log_prob.isnan().any():
        #     raise ValueError("Log probability contains NaN values. Check input parameters.")
        
        # if log_prob.isinf().any():
        #     raise ValueError("Log probability contains infinite values. Check input parameters.")
        
        # print(f"Gompertz log_prob: {log_prob.shape}, alphas: {self.alphas.shape}, betas: {self.betas.shape}")
        return log_prob

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """
        Generates a sample_shape amount of samples from the Gompertz distribution.
        
        Uses inverse CDF method.
        CDF: F(x) = 1 - exp(-(alpha/beta)*(exp(beta*x) - 1))
        Inverse CDF: x = (1/beta) * log(1 - (beta/alpha) * log(1 - U))
        where U ~ Uniform(0,1)
        """
        # Create a tensor of uniform random numbers U in (0, 1)
        # Ensure U is not exactly 0 or 1 to avoid log(0) or log(negative)
        shape = sample_shape + self.batch_shape
        u = torch.rand(shape, dtype=self.alphas.dtype, device=self.alphas.device)
        u = u.clamp(min=1e-7, max=1-1e-7) # Clamp to avoid numerical issues with log(0) and log(1)
        
        # Calculate intermediate terms for inverse CDF
        # Ensure alpha and beta are positive
        safe_alphas = self.alphas.clamp(min=1e-10)
        safe_betas = self.betas.clamp(min=1e-10)

        # Term log(1 - U): Use torch.log1p for numerical stability when 1-U is close to 1 (U close to 0)
        log_one_minus_u = torch.log1p(-u) # This is log(1-u)
        
        # Term (beta/alpha) * log(1 - U)
        # Since log(1-U) is negative, this whole term will be negative.
        beta_over_alpha_log_term = (safe_betas / safe_alphas) * log_one_minus_u
        
        # Term 1 - (beta/alpha) * log(1 - U)
        # Since beta_over_alpha_log_term is negative, this term will be 1 - (negative number) = 1 + (positive number)
        # So it will always be >= 1.
        arg_to_outer_log = 1 - beta_over_alpha_log_term
        
        # Calculate log(arg_to_outer_log)
        # No clamping needed for arg_to_outer_log as it is >= 1, so log will be >= 0.
        log_arg_to_outer_log = torch.log(arg_to_outer_log)
        
        # Final sample x = (1/beta) * log(1 - (beta/alpha) * log(1 - U))
        samples = (1.0 / safe_betas) * log_arg_to_outer_log

        return samples


class GompertzMixtureDistribution(TransformedDistribution):
    """
    Mixture of Gompertz distributions for normalized log-inter-event times.

    We model it as:
        u ~ Mixture of Gompertz(alphas, betas)
        t = exp(u * std_log_inter_time + mean_log_inter_time)

    Args:
        log_alphas: Logarithms of scale parameters for component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_betas: Logarithms of shape parameters for component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Mean of log inter-event times (for normalization)
        std_log_inter_time: Std of log inter-event times (for normalization)
        num_samples: Number of samples for Monte Carlo entropy estimation
    """
    def __init__(
        self,
        log_alphas: torch.Tensor,
        log_betas: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        num_samples: int = 1_000,
    ):
        # Convert parameters to positive domain
        # alphas = clamp_preserve_gradients(log_alphas.exp(), 0.05, 100)
        # betas = clamp_preserve_gradients(log_betas.exp(), 0.05, 10)
        alphas = log_alphas.exp()
        betas = log_betas.exp()

        # Create mixture distribution
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = GompertzDistribution(alphas=alphas, betas=betas)
        base_dist = D.MixtureSameFamily(mixture_dist, component_dist, validate_args=False)
        
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
        self.log_alphas = log_alphas
        self.log_betas = log_betas
        self.log_weights = log_weights
        self.num_samples = num_samples

    @property
    def mean(self) -> torch.Tensor:
        """Compute expected value of normalized log-inter-event times."""
        # Use Monte Carlo sampling to estimate mean
        with torch.no_grad():
            samples = self.base_dist.sample((self.num_samples,))  # Sample 1000 times
            return samples.mean(0)

    def entropy(self) -> torch.Tensor:
        """
        Compute entropy using Monte Carlo sampling.
        
        The entropy is computed as:
            H(Y) = H(X) - log(std) - E[log(X)]
        where:
            X ~ Mixture of Gompertz
            Y = (log(X) - mean) / std
        """
        with torch.no_grad():
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

class GompertzMix(RecurrentTPP):
    """
    RNN-based TPP model using a mixture of Gompertz distributions for inter-event times.

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
        layers.append(nn.Linear(input_size, 3 * self.num_mix_components))
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
        Get Gompertz mixture distribution given context.
        
        Args:
            context: History embedding, shape (batch_size, context_size)
            
        Returns:
            dist: Mixture distribution for normalized log-inter-event times
        """
        # Get raw parameters from MLP
        raw_params = self.mlp(context)
        # Split into alpha, beta, and weight parameters
        log_alphas = raw_params[..., :self.num_mix_components]
        log_betas = raw_params[..., self.num_mix_components:2*self.num_mix_components]
        log_weights = raw_params[..., 2*self.num_mix_components:]
        
        # Clamp parameters for stability
        log_alphas = clamp_preserve_gradients(log_alphas, -3, 3)
        log_betas = clamp_preserve_gradients(log_betas, -3, 3)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        
        return GompertzMixtureDistribution(
            log_alphas=log_alphas,
            log_betas=log_betas,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time,
            num_samples=self.num_samples
        )

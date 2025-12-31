import torch
import torch.nn as nn
from scipy.optimize import brentq

import torch.distributions as D
from dpp.distributions import Normal, MixtureSameFamily, TransformedDistribution
from dpp.utils import clamp_preserve_gradients
from .recurrent_tpp import RecurrentTPP


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
    ):
        self.logits = log_weights
        self.locs = locs
        self.log_scales = log_scales
        mixture_dist = D.Categorical(logits=log_weights)
        
        component_dist = D.LogNormal(loc=locs, scale=self.log_scales.exp()) # mean and standard deviation
        MLN = MixtureSameFamily(mixture_dist, component_dist)
        transforms = []
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        
        super().__init__(MLN, transforms)


class LogNormMix(RecurrentTPP):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
        self,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        num_mix_components: int = 16,
        hidden_sizes = [16],
        num_processes: int = 1,
        mean_log_inter_time_source: float = 0.0,
        std_log_inter_time_source: float = 1.0,
        context_extractor: str = "mlp",
        activation_func: str = "Tanh",
        history_length: int = 16,
    ):
        super().__init__(
            num_marks=num_marks,
            mean_log_inter_time=mean_log_inter_time,
            std_log_inter_time=std_log_inter_time,
            context_size=context_size,
            num_processes=num_processes,
            mean_log_inter_time_source=mean_log_inter_time_source,
            std_log_inter_time_source=std_log_inter_time_source,
            context_extractor=context_extractor,
            history_length=history_length,
        )
        self.num_mix_components = num_mix_components
        # self.linear = nn.Linear(self.context_size, 3 * self.num_mix_components)
        # Allow arbitrary number of hidden layers in the MLP
        layers = [getattr(nn, activation_func)()]
        input_size = self.context_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(getattr(nn, activation_func)())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 3 * self.num_mix_components))
        self.mlp = nn.Sequential(*layers)

        # Initialization function
        def init_mlp_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        # Apply initialization
        self.mlp.apply(init_mlp_weights)

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size,)

        """
        raw_params = self.mlp(context)  # (batch_size, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components] # mean of the mixture components

        # log_scales: logarithm of standard deviation
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5, 3) #-5,3 | -2, 0
        # locs = clamp_preserve_gradients(locs, -8, 2) # -8, 2
        log_weights = torch.log_softmax(log_weights, dim=-1)
        
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
        )
    

# --- Example Usage ---
if __name__ == '__main__':
    import torch
    from torch.distributions import Normal, Categorical
    from torch.distributions.transforms import AffineTransform

    # Create a MixtureSameFamily of Normal distributions
    # Mixture components:
    # Component 0: Normal(loc=-2, scale=1)
    # Component 1: Normal(loc=0, scale=0.5)
    # Component 2: Normal(loc=3, scale=1.5)

    # Component parameters should be batched along the last dimension for MixtureSameFamily
    locs = torch.tensor([-2.0, 0.0, 3.0])
    scales = torch.tensor([1.0, 0.5, 1.5])
    
    # The component_distribution itself needs to be a batched distribution
    # where the batch dimension corresponds to the components.
    component_dist = Normal(loc=locs, scale=scales)
    
    # Mixture weights
    mix_logits = torch.tensor([0.5, 0.3, 0.2]) # Weights for components 0, 1, 2
    mixture_dist = Categorical(logits=mix_logits)

    # Instantiate the custom MixtureSameFamily
    mixture_same_family = MixtureSameFamily(
        mixture_distribution=mixture_dist,
        component_distribution=component_dist
    )

    print("MixtureSameFamily instance created.")
    print("Batch shape:", mixture_same_family.batch_shape)
    print("Event shape:", mixture_same_family.event_shape) # Should be torch.Size([])

    # Test log_cdf
    test_x = torch.tensor([-5.0, -2.0, 0.0, 3.0, 5.0])
    log_cdf_val = mixture_same_family.log_cdf(test_x)
    print("\nLog CDF for test_x:", log_cdf_val)
    print("CDF for test_x:", torch.exp(log_cdf_val))

    # Test log_survival_function
    log_sf_val = mixture_same_family.log_survival_function(test_x)
    print("\nLog Survival Function for test_x:", log_sf_val)
    print("Survival Function for test_x:", torch.exp(log_sf_val))

    # Test icdf
    probabilities_to_test = torch.tensor([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
    print("\nProbabilities to test for ICDF:", probabilities_to_test)

    quantiles = mixture_same_family.icdf(probabilities_to_test)
    print("Calculated Quantiles (ICDF):", quantiles)

    # Verify ICDF by plugging quantiles back into CDF
    # For numerical stability, check if log_cdf(icdf(p)) is close to log(p)
    log_p_reconstructed = mixture_same_family.log_cdf(quantiles)
    print("Reconstructed Log Probabilities (from CDF of Quantiles):", log_p_reconstructed)
    print("Original Log Probabilities:", torch.log(probabilities_to_test))
    
    # Check the absolute difference
    abs_diff = torch.abs(torch.exp(log_p_reconstructed) - probabilities_to_test)
    print("Absolute difference (reconstructed_cdf - original_p):", abs_diff)
    print("Max absolute difference:", abs_diff.max().item())
    
    assert abs_diff.max().item() < 1e-5, "ICDF verification failed: values not close enough"
    print("\nICDF verification passed!")


    # Example with a batched mixture distribution (multiple mixtures)
    print("\n--- Testing with Batched MixtureSameFamily ---")
    
    # Two mixtures
    batch_locs = torch.tensor([[-2.0, 0.0, 3.0], [5.0, 7.0, 9.0]]) # Shape (batch_dim, num_components)
    batch_scales = torch.tensor([[1.0, 0.5, 1.5], [0.8, 1.2, 0.7]]) # Shape (batch_dim, num_components)
    batch_component_dist = Normal(loc=batch_locs, scale=batch_scales)

    batch_mix_logits = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]) # Shape (batch_dim, num_components)
    batch_mixture_dist = Categorical(logits=batch_mix_logits)

    batched_mixture_same_family = MixtureSameFamily(
        mixture_distribution=batch_mixture_dist,
        component_distribution=batch_component_dist
    )
    print("Batched MixtureSameFamily instance created.")
    print("Batch shape:", batched_mixture_same_family.batch_shape) # Should be (2,)
    print("Event shape:", batched_mixture_same_family.event_shape) # Should be torch.Size([])

    batch_probabilities = torch.tensor([[0.25, 0.5, 0.75], [0.1, 0.5, 0.9]]) # Shape (2, 3)
    print("\nBatch probabilities to test for ICDF:\n", batch_probabilities)

    batch_quantiles = batched_mixture_same_family.icdf(batch_probabilities)
    print("Calculated Quantiles (Batched ICDF):\n", batch_quantiles)

    batch_log_p_reconstructed = batched_mixture_same_family.log_cdf(batch_quantiles)
    print("Reconstructed Log Probabilities (Batched):\n", torch.exp(batch_log_p_reconstructed))
    print("Original Log Probabilities (Batched):\n", batch_probabilities)
    
    batch_abs_diff = torch.abs(torch.exp(batch_log_p_reconstructed) - batch_probabilities)
    print("Max absolute difference (batched):", batch_abs_diff.max().item())
    assert batch_abs_diff.max().item() < 1e-5, "Batched ICDF verification failed!"
    print("\nBatched ICDF verification passed!")

    # Test with a scalar probability input to a batched mixture
    scalar_p = torch.tensor(0.5)
    scalar_quantiles = batched_mixture_same_family.icdf(scalar_p)
    print("\nScalar probability (0.5) for batched mixture:", scalar_quantiles)
    # The output should have the mixture's batch shape: (2,)
    assert scalar_quantiles.shape == batched_mixture_same_family.batch_shape
    assert torch.abs(torch.exp(batched_mixture_same_family.log_cdf(scalar_quantiles)) - scalar_p).max().item() < 1e-5
    print("Scalar input to batched ICDF verification passed!")
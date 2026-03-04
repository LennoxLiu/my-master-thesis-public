import torch

from torch.distributions import Normal as TorchNormal

from dpp.utils import clamp_preserve_gradients


class Normal(TorchNormal):
    def log_cdf(self, x):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)

    def icdf(self, value):
        """
        Computes the Inverse Cumulative Distribution Function (ICDF) for the Normal distribution.

        This method leverages the analytical `icdf` (quantile) implementation
        provided by `torch.distributions.Normal`. Input probabilities `value`
        are clamped to be strictly within (0, 1) to ensure numerical stability
        for the inverse function, similar to how `log_cdf` handles its inputs.

        Args:
            value (torch.Tensor): Probability values (0 < value < 1) for which
                                  to find the quantiles. Can be a scalar or a tensor.

        Returns:
            torch.Tensor: The corresponding quantile values.
        """
        # Ensure value is strictly between 0 and 1 for the inverse CDF.
        # Clamping helps prevent issues with log(0) or log(1) if this icdf
        # were to be used in conjunction with log-probabilities,
        # or if the base icdf has issues with exact 0 or 1 inputs.
        clamped_value = clamp_preserve_gradients(value, 1e-7, 1 - 1e-7)
        
        # The base `torch.distributions.Normal` already provides a robust icdf method.
        # This method is also known as `quantile` in PyTorch distributions.
        return super().icdf(clamped_value)
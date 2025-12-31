from torch.distributions import TransformedDistribution as TorchTransformedDistribution


class TransformedDistribution(TorchTransformedDistribution):
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=validate_args)
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)

    def icdf(self, value):
        """
        Computes the Inverse Cumulative Distribution Function (ICDF) for the
        transformed distribution.

        This method assumes:
        1. The base_distribution has an `icdf` method.
        2. All transforms have a `forward` method which is the inverse of `inv`.
           (This is standard for torch.distributions.transforms).

        Args:
            value (torch.Tensor): Probability values (0 < value < 1) for which
                                  to find the quantiles. Can be a scalar or a tensor.

        Returns:
            torch.Tensor: The corresponding quantile values in the transformed space.
        """
        if not ((value > 0) & (value < 1)).all():
            raise ValueError("Input `value` (probability) must be between 0 and 1 (exclusive).")

        # Handle the sign of the overall transformation
        # If sign is -1, a probability `p` in the transformed space corresponds to
        # `1 - p` in the base distribution's inverse-transformed variable.
        # Example: if X = -Z, then P(X <= x) = P(-Z <= x) = P(Z >= -x) = 1 - P(Z <= -x)
        # So, F_X(x) = 1 - F_Z(-x). If we want F_X(x) = p, then 1 - F_Z(-x) = p,
        # F_Z(-x) = 1 - p. Then -x = F_Z^{-1}(1-p), so x = -F_Z^{-1}(1-p).
        # Generalizing: apply icdf to `value` or `1 - value` based on `self.sign`.

        # Prepare probability for base distribution's ICDF
        base_dist_p = value if self.sign == 1 else (1 - value)

        # 1. Get the quantile from the base distribution
        # This gives us x_base: the value in the base distribution's domain
        x_base = self.base_dist.icdf(base_dist_p)

        # 2. Apply all transforms in their forward direction
        # This takes x_base to the transformed space
        x_transformed = x_base
        for transform in self.transforms:
            x_transformed = transform(x_transformed)

        return x_transformed


if __name__ == "__main__":
    import torch
    from torch.distributions import Normal, LogNormal
    import torch.distributions.transforms as T
    # --- Example Usage ---

    # Example 1: Log-Normal distribution (Normal -> Exp)
    print("--- Example 1: Log-Normal (Normal + ExpTransform) ---")
    normal_dist = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
    exp_transform = T.ExpTransform()
    log_normal_td = TransformedDistribution(normal_dist, [exp_transform])

    # Test ICDF for Log-Normal
    probabilities = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    icdf_values = log_normal_td.icdf(probabilities)
    print(f"Probabilities: {probabilities}")
    print(f"Log-Normal ICDF values: {icdf_values}")

    # Verify with torch.distributions.LogNormal's ICDF
    log_normal_builtin = LogNormal(loc=normal_dist.loc, scale=normal_dist.scale)
    icdf_builtin_values = log_normal_builtin.icdf(probabilities)
    print(f"Built-in LogNormal ICDF values: {icdf_builtin_values}")
    assert torch.allclose(icdf_values, icdf_builtin_values, atol=1e-6)
    print("Matches built-in LogNormal ICDF!")
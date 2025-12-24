"""Utilities for normalizing observations and rewards."""

import jax.numpy as jnp
from chex import Array

from dopamax.typing import Observation


class RunningStats:
    """Maintains running statistics (mean and variance) for normalization.

    This is useful for normalizing observations or rewards during training.
    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32):
        """Initialize running statistics.

        Args:
            shape: The shape of the statistics to track.
            dtype: The dtype of the statistics.
        """
        self._shape = shape
        self._dtype = dtype
        self._count = 0
        self._mean = jnp.zeros(shape, dtype=dtype)
        self._var = jnp.ones(shape, dtype=dtype)

    @property
    def count(self) -> int:
        """The number of samples seen."""
        return self._count

    @property
    def mean(self) -> Array:
        """The current mean."""
        return self._mean

    @property
    def std(self) -> Array:
        """The current standard deviation."""
        return jnp.sqrt(self._var)

    def update(self, x: Array) -> None:
        """Update statistics with a new sample.

        Args:
            x: A new sample to incorporate into the statistics.
        """
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._var += delta * delta2

    def normalize(self, x: Array, epsilon: float = 1e-8) -> Array:
        """Normalize a value using the current statistics.

        Args:
            x: The value to normalize.
            epsilon: Small value to prevent division by zero.

        Returns:
            The normalized value: (x - mean) / (std + epsilon)
        """
        return (x - self._mean) / (self.std + epsilon)

    def denormalize(self, x: Array) -> Array:
        """Denormalize a value using the current statistics.

        Args:
            x: The normalized value.

        Returns:
            The denormalized value: x * std + mean
        """
        return x * self.std + self._mean


def normalize_observations(observations: Observation, stats: dict[str, RunningStats]) -> Observation:
    """Normalize observations using running statistics.

    Args:
        observations: The observations to normalize. Can be a dict of arrays or a single array.
        stats: A dictionary mapping observation keys to RunningStats objects.

    Returns:
        The normalized observations with the same structure as the input.
    """
    if isinstance(observations, dict):
        normalized = {}
        for key, value in observations.items():
            if key in stats:
                normalized[key] = stats[key].normalize(value)
            else:
                normalized[key] = value
        return normalized
    else:
        # Single array - use default key
        if "observation" in stats:
            return stats["observation"].normalize(observations)
        return observations


def normalize_rewards(rewards: Array, mean: float, std: float, epsilon: float = 1e-8) -> Array:
    """Normalize rewards using given statistics.

    Args:
        rewards: The rewards to normalize.
        mean: The mean reward value.
        std: The standard deviation of rewards.
        epsilon: Small value to prevent division by zero.

    Returns:
        The normalized rewards: (rewards - mean) / (std + epsilon)
    """
    return (rewards - mean) / (std + epsilon)


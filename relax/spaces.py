from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey, Array, Scalar, ArrayTree


class Space(ABC):
    """Abstract class that defines observation and action spaces.

    Args:
        dtype: The dtype of the space.
    """

    def __init__(self, dtype: jnp.dtype):
        self._dtype = dtype

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """The shape of the space."""
        pass

    @abstractmethod
    def sample(self, key: PRNGKey) -> ArrayTree:
        """Sample a random element from the space.

        Args:
            key: A PRNGKey used to generate the sample.

        Returns:
            A random element from the space.
        """
        pass

    @abstractmethod
    def contains(self, item: ArrayTree) -> bool:
        """Check if an item is contained in the space.

        Args:
            item: The item to check.

        Returns:
            True if the item is contained in the space, False otherwise.
        """
        pass

    def __contains__(self, item: ArrayTree) -> bool:
        return self.contains(item)


class Discrete(Space):
    """A discrete space with finite elements.

    This space represents the integers from 0 to n - 1.

    Args:
        n: The number of discrete elements in the space.
        dtype: The dtype of the space.
    """

    def __init__(self, n: int, dtype: jnp.dtype = jnp.int32):
        super().__init__(dtype)

        assert n > 0, "Discrete space must have n > 0."

        self._n = n

    @property
    def shape(self) -> Tuple[int, ...]:
        return ()

    def sample(self, key: PRNGKey) -> ArrayTree:
        return jax.random.choice(key, self._n).astype(self.dtype)

    def contains(self, item: ArrayTree) -> bool:
        return jnp.ndim(item) == 0 and 0 <= item < self._n

    @property
    def n(self) -> int:
        return self._n

    def __repr__(self):
        return f"Discrete({self._n})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Discrete) and self.n == other.n


class Box(Space):
    """A (possibly unbounded) box in R^n.

    Args:
        low: The lower bounds of the intervals.
        high: The upper bounds of the intervals.
        shape: The shape of the space. If None, the shape is inferred from low and high.
        dtype: The dtype of the space.
    """

    def __init__(
        self,
        low: Array | Scalar,
        high: Array | Scalar,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__(dtype)

        low = jnp.asarray(low)
        high = jnp.asarray(high)

        if shape is None:
            shape = jnp.broadcast_shapes(low.shape, high.shape)

        low = jnp.broadcast_to(low, shape)
        high = jnp.broadcast_to(high, shape)

        assert low.shape == high.shape, "low and high must have the same shape."

        self._low = low.astype(dtype)
        self._high = high.astype(dtype)

    @property
    def low(self) -> Array | Scalar:
        return self._low

    @property
    def high(self) -> Array | Scalar:
        return self._high

    @property
    def bounded_below(self) -> bool:
        return -jnp.inf < self.low

    @property
    def bounded_above(self) -> bool:
        return jnp.inf > self.high

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._low.shape

    def sample(self, key: PRNGKey) -> ArrayTree:
        unbounded = ~self.bounded_below & ~self.bounded_above
        upper_bounded = ~self.bounded_below & self.bounded_above
        lower_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        sample_unbounded = jax.random.normal(key, shape=self.shape)
        sample_lower_bounded = jax.random.exponential(key, shape=self.shape) + self.low
        sample_upper_bounded = -jax.random.exponential(key, shape=self.shape) + self.high
        sample_bounded = jax.random.uniform(key, shape=self.shape, minval=self.low, maxval=self.high)

        sample = jnp.sum(
            jnp.stack(
                [
                    sample_unbounded * unbounded,
                    sample_lower_bounded * lower_bounded,
                    sample_upper_bounded * upper_bounded,
                    sample_bounded * bounded,
                ]
            ),
            axis=0,
        )

        if jnp.dtype(self.dtype).kind == "i":
            sample = jnp.floor(sample)

        return sample.astype(self.dtype)

    def contains(self, item: ArrayTree) -> bool:
        return bool(
            jnp.can_cast(item.dtype, self.dtype)
            and item.shape == self.shape
            and jnp.all(item >= self.low)
            and jnp.all(item <= self.high)
        )

    def __repr__(self):
        return f"Box(low={self.low}, high={self.high})"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Box)
            and jnp.array_equal(self.low == other.low)
            and jnp.array_equal(self.high == other.high)
        )

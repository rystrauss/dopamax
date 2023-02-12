import jax

from dopamax.spaces import Discrete, Box
import jax.numpy as jnp
import pytest


def test_discrete():
    space = Discrete(10)

    assert space.shape == ()
    assert space.dtype == jnp.int32
    assert space.n == 10

    for i in range(10):
        assert i in space
        assert space.contains(i)

        i = jnp.array(i, dtype=jnp.int32)
        assert i in space
        assert space.contains(i)

    assert 10 not in space
    assert -1 not in space


@pytest.mark.parametrize(
    "low,high,shape",
    [
        (0, 5, ()),
        (0, 5, (1,)),
        (-jnp.inf, 100, (200,)),
        (-jnp.inf, jnp.inf, (20, 50)),
        (-10, 10, None),
        (jnp.array([1, 2, 3, 4, 5]), 10, None),
    ],
)
def test_box(low, high, shape):
    space = Box(low, high, shape)

    if shape is not None:
        assert space.shape == shape
    else:
        assert space.shape == jnp.broadcast_shapes(jnp.asarray(low).shape, jnp.asarray(high).shape)

    assert space.dtype == jnp.float32

    assert jnp.array_equal(space.low, jnp.broadcast_to(low, space.shape))
    assert jnp.array_equal(space.high, jnp.broadcast_to(high, space.shape))

    for i in range(100):
        sample = space.sample(jax.random.PRNGKey(i))
        assert sample.shape == space.shape
        assert sample in space

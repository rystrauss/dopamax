import chex
import jax.numpy as jnp
import jax.random

from dopamax.rollouts import create_minibatches


def test_create_minibatches():
    rollout_data = {
        "a": jnp.repeat(jnp.expand_dims(jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1), 2, 1),
        "b": jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 10,
    }

    key = jax.random.PRNGKey(9)

    minibatches = create_minibatches(key, rollout_data, 2)

    chex.assert_shape(minibatches["a"], (5, 2, 2))
    chex.assert_shape(minibatches["b"], (5, 2))

    for i in range(5):
        assert jnp.all(minibatches["a"][i] * 10 == jnp.expand_dims(minibatches["b"][i], 1))

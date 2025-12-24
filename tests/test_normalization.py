"""Tests for normalization utilities."""

import jax
import jax.numpy as jnp

from dopamax.utils.normalization import (
    RunningStats,
    normalize_observations,
    normalize_rewards,
)


def test_running_stats():
    """Test RunningStats functionality."""
    stats = RunningStats(shape=(4,), dtype=jnp.float32)

    # Generate some test data
    key = jax.random.PRNGKey(42)
    data = jax.random.normal(key, shape=(100, 4)) * 2.0 + 1.0

    # Update stats with data
    for sample in data:
        stats.update(sample)

    assert stats.count == 100

    # Test normalization
    normalized = stats.normalize(data[0])
    assert normalized.shape == (4,)

    # Test denormalization
    denormalized = stats.denormalize(normalized)
    assert jnp.allclose(denormalized, data[0], atol=1e-5)


def test_normalize_observations_dict():
    """Test normalizing dictionary observations."""
    stats = {
        "observation": RunningStats(shape=(4,), dtype=jnp.float32),
        "other": RunningStats(shape=(2,), dtype=jnp.float32),
    }

    key = jax.random.PRNGKey(42)
    obs = {
        "observation": jax.random.normal(key, shape=(4,)),
        "other": jax.random.normal(key, shape=(2,)),
        "unused": jax.random.normal(key, shape=(3,)),
    }

    # Update stats
    stats["observation"].update(obs["observation"])
    stats["other"].update(obs["other"])

    normalized = normalize_observations(obs, stats)

    assert "observation" in normalized
    assert "other" in normalized
    assert "unused" in normalized
    assert jnp.allclose(normalized["unused"], obs["unused"])


def test_normalize_observations_array():
    """Test normalizing array observations."""
    stats = {"observation": RunningStats(shape=(4,), dtype=jnp.float32)}

    key = jax.random.PRNGKey(42)
    obs = jax.random.normal(key, shape=(4,))

    stats["observation"].update(obs)
    normalized = normalize_observations(obs, stats)

    assert normalized.shape == (4,)


def test_normalize_rewards():
    """Test reward normalization."""
    key = jax.random.PRNGKey(42)
    rewards = jax.random.normal(key, shape=(100,)) * 2.0 + 1.0

    mean = jnp.mean(rewards)
    std = jnp.std(rewards)

    normalized = normalize_rewards(rewards, mean, std)

    assert jnp.allclose(jnp.mean(normalized), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.std(normalized), 1.0, atol=1e-5)


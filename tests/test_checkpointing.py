"""Tests for checkpointing functionality."""

import tempfile
from pathlib import Path

import jax
import pytest

from dopamax.agents.utils import get_agent_cls
from dopamax.environments import make_env
from dopamax.utils.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    load_params,
    save_checkpoint,
    save_params,
)


def test_save_load_checkpoint():
    """Test saving and loading checkpoints."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = save_checkpoint(tmpdir, train_state, step=10, metadata={"test": "value"})

        assert checkpoint_path.exists()

        loaded_state, step, metadata = load_checkpoint(checkpoint_path)

        assert step == 10
        assert metadata["test"] == "value"
        assert loaded_state.train_step == train_state.train_step
        assert loaded_state.total_timesteps == train_state.total_timesteps


def test_find_latest_checkpoint():
    """Test finding the latest checkpoint."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save multiple checkpoints
        save_checkpoint(tmpdir, train_state, step=5)
        save_checkpoint(tmpdir, train_state, step=10)
        save_checkpoint(tmpdir, train_state, step=15)

        latest = find_latest_checkpoint(tmpdir)

        assert latest is not None
        assert "checkpoint_step_15" in latest.name


def test_save_load_params():
    """Test saving and loading parameters."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = Path(tmpdir) / "params.pkl"
        save_params(params_path, train_state.params)

        assert params_path.exists()

        loaded_params = load_params(params_path)

        # Check that the structure matches
        assert jax.tree.structure(loaded_params) == jax.tree.structure(train_state.params)


def test_checkpoint_nonexistent_file():
    """Test that loading non-existent checkpoint raises appropriate error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent = Path(tmpdir) / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            load_checkpoint(nonexistent)

        with pytest.raises(FileNotFoundError):
            load_params(nonexistent)


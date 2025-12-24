"""Integration tests for full training loops."""

import jax
import pytest

from dopamax.agents.utils import get_agent_cls
from dopamax.environments import make_env


@pytest.mark.parametrize("agent_name", ["PPO", "DQN"])
def test_ppo_dqn_training_loop(agent_name):
    """Test that PPO and DQN can complete a short training loop."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls(agent_name)
    config = agent_cls.default_config()
    config.rollout_fragment_length = 32
    config.minibatch_size = 16
    config.num_epochs = 1
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    # Run a few training steps
    for _ in range(3):
        train_state, metrics = agent.train_step(train_state)
        assert train_state.train_step > 0
        assert "loss" in metrics or "policy_loss" in metrics or "value_loss" in metrics


def test_sac_training_loop():
    """Test that SAC can complete a short training loop."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("SAC")
    config = agent_cls.default_config()
    config.learning_starts = 10
    config.batch_size = 16
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    # Run a few training steps
    for _ in range(15):
        train_state, metrics = agent.train_step(train_state)
        assert train_state.train_step > 0
        if train_state.train_step > config.learning_starts:
            assert "actor_loss" in metrics or "critic_loss" in metrics


def test_agent_compute_action_consistency():
    """Test that compute_action works consistently across agents."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)
    obs = env.observation_space.sample(key)

    # Test deterministic action
    action_det = agent.compute_action(train_state.params, key, obs, deterministic=True)
    assert action_det.shape == env.action_space.shape

    # Test stochastic action
    action_stoch = agent.compute_action(train_state.params, key, obs, deterministic=False)
    assert action_stoch.shape == env.action_space.shape


def test_training_state_progression():
    """Test that training state correctly tracks progress."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    config.rollout_fragment_length = 32
    config.minibatch_size = 16
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    initial_step = train_state.train_step
    initial_timesteps = train_state.total_timesteps
    initial_episodes = train_state.total_episodes

    train_state, _ = agent.train_step(train_state)

    assert train_state.train_step == initial_step + 1
    assert train_state.total_timesteps > initial_timesteps
    assert train_state.total_episodes >= initial_episodes


def test_edge_case_empty_buffer():
    """Test that agents handle edge cases gracefully."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("DQN")
    config = agent_cls.default_config()
    config.learning_starts = 5
    config.batch_size = 4
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    # Run a few steps before learning starts (buffer might be small)
    for _ in range(3):
        train_state, metrics = agent.train_step(train_state)
        # Should not crash even with small buffer
        assert isinstance(metrics, dict)


def test_multi_episode_rollout():
    """Test that rollouts can span multiple episodes."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    config.rollout_fragment_length = 192  # Longer than typical episode, divisible by 32
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(42)
    train_state = agent.initial_train_state(key, key)

    # Should handle rollouts that span multiple episodes
    train_state, metrics = agent.train_step(train_state)
    assert train_state.total_episodes > 0


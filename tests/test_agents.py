"""Unit tests for RL agents."""

import jax
import pytest

from dopamax.agents.utils import get_agent_cls
from dopamax.environments import make_env


def test_ppo_initialization():
    """Test that PPO agent initializes correctly."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    assert agent.observation_space == env.observation_space
    assert agent.action_space == env.action_space


def test_ppo_config_validation():
    """Test that PPO validates configuration correctly."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    config.rollout_fragment_length = 100
    config.minibatch_size = 33  # Not divisible

    with pytest.raises(ValueError, match="must be divisible"):
        agent_cls(env, config)


def test_dqn_initialization():
    """Test that DQN agent initializes correctly."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("DQN")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    assert agent.observation_space == env.observation_space
    assert agent.action_space == env.action_space


def test_dqn_discrete_action_space_requirement():
    """Test that DQN requires discrete action spaces."""
    from dopamax.environments.brax.ant import Ant

    env = Ant()  # Continuous action space
    agent_cls = get_agent_cls("DQN")
    config = agent_cls.default_config()

    with pytest.raises(ValueError, match="discrete action spaces"):
        agent_cls(env, config)


def test_dqn_dueling_validation():
    """Test that DQN validates dueling network configuration."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("DQN")
    config = agent_cls.default_config()
    config.dueling = True
    config.final_hidden_units = ()  # Empty - should fail

    with pytest.raises(ValueError, match="at least one hidden layer"):
        agent_cls(env, config)


def test_ddpg_continuous_action_space_requirement():
    """Test that DDPG requires continuous action spaces."""
    env = make_env("gymnax:CartPole-v1")  # Discrete action space
    agent_cls = get_agent_cls("DDPG")
    config = agent_cls.default_config()

    with pytest.raises(ValueError, match="continuous.*action spaces"):
        agent_cls(env, config)


def test_sac_initialization():
    """Test that SAC agent initializes correctly for both discrete and continuous spaces."""
    # Discrete
    env_discrete = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("SAC")
    config = agent_cls.default_config()
    agent_discrete = agent_cls(env_discrete, config)
    assert agent_discrete.observation_space == env_discrete.observation_space

    # Continuous
    from dopamax.environments.brax.ant import Ant

    env_continuous = Ant()
    agent_continuous = agent_cls(env_continuous, config)
    assert agent_continuous.observation_space == env_continuous.observation_space


def test_sac_config_validation():
    """Test that SAC validates configuration correctly."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("SAC")
    config = agent_cls.default_config()
    config.tau = 1.5  # Invalid

    with pytest.raises(ValueError, match="tau must be in"):
        agent_cls(env, config)


def test_agent_compute_action():
    """Test that agents can compute actions."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    key = jax.random.PRNGKey(0)
    train_state = agent.initial_train_state(key, key)
    obs = env.observation_space.sample(key)

    action = agent.compute_action(train_state.params, key, obs, deterministic=True)
    assert action.shape == env.action_space.shape


def test_agent_initial_train_state():
    """Test that agents can create initial train states."""
    env = make_env("gymnax:CartPole-v1")
    agent_cls = get_agent_cls("PPO")
    config = agent_cls.default_config()
    agent = agent_cls(env, config)

    consistent_key = jax.random.PRNGKey(0)
    divergent_key = jax.random.PRNGKey(1)

    train_state = agent.initial_train_state(consistent_key, divergent_key)
    assert train_state.train_step == 0
    assert train_state.total_timesteps == 0
    assert train_state.total_episodes == 0


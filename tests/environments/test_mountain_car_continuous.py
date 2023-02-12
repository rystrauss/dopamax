import chex
import gymnasium as gym
import jax
import numpy as np
import pytest
from dm_env import StepType

from dopamax.environments.mountain_car_continuous import MountainCarContinuous


def test_mountain_car_continuous():
    key = jax.random.PRNGKey(0)

    jax_env = MountainCarContinuous()
    gym_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    time_step, state = jax_env.reset(key)
    chex.assert_trees_all_equal(time_step.observation, state.to_obs())

    gym_env.reset()
    gym_env.unwrapped.state = (state.position, state.velocity)

    assert gym_env.unwrapped.state[0] == pytest.approx(state.position, rel=0.001)
    assert gym_env.unwrapped.state[1] == pytest.approx(state.velocity, rel=0.001)

    for _ in range(300):
        gym_render = gym_env.render()
        jax_render = jax_env.render(state)

        chex.assert_trees_all_equal(gym_render, jax_render)
        chex.assert_shape((gym_render, jax_render), jax_env.render_shape)

        action = jax_env.action_space.sample(key)
        time_step, state = jax_env.step(key, state, action)

        gym_obs, gym_reward, gym_terminated, gym_truncated, _ = gym_env.step(np.array(action))

        assert gym_reward == pytest.approx(time_step.reward, rel=0.0001)
        assert (gym_terminated or gym_truncated) == bool(time_step.step_type == StepType.LAST)
        assert gym_env.unwrapped.state[0] == pytest.approx(state.position, rel=0.001)
        assert gym_env.unwrapped.state[1] == pytest.approx(state.velocity, rel=0.001)
        chex.assert_trees_all_close(gym_obs, time_step.observation, rtol=0.001)

        key, _ = jax.random.split(key)

        if gym_terminated or gym_truncated:
            time_step, state = jax_env.reset(key)
            gym_env.reset()
            gym_env.unwrapped.state = (state.position, state.velocity)

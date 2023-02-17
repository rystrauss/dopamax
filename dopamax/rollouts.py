from typing import Tuple, Dict, Protocol

import einops
import haiku as hk
import jax
import jax.numpy as jnp
from chex import PRNGKey, ArrayTree
from dm_env import StepType

from dopamax.environments.environment import Environment, EnvState, TimeStep
from dopamax.typing import Observation, Action


class SampleBatch(dict):
    OBSERVATION = "observation"
    NEXT_OBSERVATION = "next_observation"
    REWARD = "reward"
    RETURN = "return"
    ADVANTAGE = "advantage"
    DISCOUNT = "discount"
    ACTION = "action"
    ACTION_LOGP = "action_logp"
    VALUE = "value"
    STEP_TYPE = "step_type"
    POLICY_STATE = "policy_state"
    NEXT_POLICY_STATE = "next_policy_state"
    VALID_MASK = "valid_mask"
    EPISODE_REWARD = "episode_reward"
    EPISODE_LENGTH = "episode_length"
    RENDER = "render"


# Type definition for a policy function that can be used in a rollout.
class PolicyFn(Protocol):
    def __call__(
        self, params: hk.Params, key: PRNGKey, observation: Observation, **kwargs
    ) -> Tuple[Action, Dict[str, ArrayTree]]:
        ...


def rollout_episode(
    env: Environment,
    policy_fn: PolicyFn,
    policy_params: hk.Params,
    key: PRNGKey,
    render: bool = False,
    **policy_fn_kwargs,
) -> SampleBatch:
    """Rollout a single episode according to the given policy.

    Args:
        env: The environment to rollout in.
        policy_fn: The policy function, which accepts the policy parameters, a PRNG key, and an observation
            and returns an action.
        policy_params: The policy parameters to feed into the policy function.
        key: A PRNG key.
        render: Whether to include environment renders in the rollout.

    Returns:
        A dictionary containing trajectory data from the rollout.
    """

    if render:
        assert env.renderable, "Environment cannot be rendered."

    def transition_fn(carry, _):
        key, time_step, env_state, valid_mask = carry

        key, step_key, reset_env_key, policy_key = jax.random.split(key, 4)

        if render:
            frame = jax.pure_callback(
                env.render,
                jax.ShapeDtypeStruct(env.render_shape, jnp.uint8),
                env_state,
                vectorized=False,
            )

        action, policy_info = policy_fn(policy_params, policy_key, time_step.observation, **policy_fn_kwargs)

        next_time_step, next_env_state = env.step(step_key, env_state, action)

        next_valid_mask = valid_mask * (1.0 - (next_time_step.step_type == StepType.LAST))

        data = {
            SampleBatch.OBSERVATION: time_step.observation,
            SampleBatch.REWARD: next_time_step.reward,
            SampleBatch.DISCOUNT: next_time_step.discount,
            SampleBatch.ACTION: action,
            SampleBatch.NEXT_OBSERVATION: next_time_step.observation,
            SampleBatch.STEP_TYPE: next_time_step.step_type,
            SampleBatch.VALID_MASK: valid_mask,
            SampleBatch.EPISODE_REWARD: next_env_state.episode_reward,
            SampleBatch.EPISODE_LENGTH: next_env_state.episode_length,
            **policy_info,
        }

        if render:
            data[SampleBatch.RENDER] = frame

        return (key, next_time_step, next_env_state, next_valid_mask), data

    env_key, scan_key = jax.random.split(key)
    time_step, env_state = env.reset(env_key)
    init = (scan_key, time_step, env_state, 1.0)
    _, rollout_data = jax.lax.scan(transition_fn, init, None, length=env.max_episode_length)

    return rollout_data


def rollout_truncated(
    env: Environment,
    num_steps: int,
    policy_fn: PolicyFn,
    policy_params: hk.Params,
    key: PRNGKey,
    time_step: TimeStep,
    env_state: EnvState,
    **policy_fn_kwargs,
) -> Tuple[SampleBatch, PRNGKey, EnvState, TimeStep]:
    """Rollout for a given number of steps, possibly over multiple episodes.

    If an episode ends before the given number of steps, the environment will be automatically reset, and the trajectory
    may include an incomplete episode if the number of steps is reached before the current episode ends.

    Args:
        env: The environment to rollout in.
        policy_fn: The policy function, which accepts the policy parameters, a PRNG key, and an observation
            and returns an action.
        policy_params: The policy parameters to feed into the policy function.
        key: A PRNG key.
        time_step: The initial time step.
        env_state: The initial environment state.

    Returns:
        rollout_data: A dictionary containing trajectory data from the rollout.
        key: The PRNG key after the final transition.
        time_step: The time step after the final transition.
        env_state: The environment state after the final transition.
    """

    def transition_fn(carry, _):
        key, time_step, env_state = carry

        key, step_key, reset_env_key, policy_key = jax.random.split(key, 4)

        action, policy_info = policy_fn(policy_params, policy_key, time_step.observation, **policy_fn_kwargs)

        next_time_step, next_env_state = env.step(step_key, env_state, action)

        data = {
            SampleBatch.OBSERVATION: time_step.observation,
            SampleBatch.REWARD: next_time_step.reward,
            SampleBatch.DISCOUNT: next_time_step.discount,
            SampleBatch.ACTION: action,
            SampleBatch.NEXT_OBSERVATION: next_time_step.observation,
            SampleBatch.STEP_TYPE: next_time_step.step_type,
            SampleBatch.EPISODE_REWARD: next_env_state.episode_reward,
            SampleBatch.EPISODE_LENGTH: next_env_state.episode_length,
            **policy_info,
        }

        next_time_step, next_env_state = jax.tree_map(
            lambda x, y: jax.lax.select(next_time_step.step_type == StepType.LAST, x, y),
            env.reset(reset_env_key),
            (next_time_step, next_env_state),
        )

        return (key, next_time_step, next_env_state), data

    init = (key, time_step, env_state)
    (final_key, final_time_step, final_env_state), rollout_data = jax.lax.scan(
        transition_fn, init, None, length=num_steps
    )

    return rollout_data, final_key, final_time_step, final_env_state


def create_minibatches(key: PRNGKey, rollout_data: SampleBatch, minibatch_size: int) -> SampleBatch:
    """Separates a rollout into minibatches.

    Args:
        key: A PRNG key used to shuffle the data.
        rollout_data: A SampleBatch containing the rollout data.
        minibatch_size: The size of each minibatch.

    Returns:
        A SampleBatch containing the minibatches.
    """
    return jax.tree_map(
        lambda x: einops.rearrange(
            jax.random.permutation(key, x, independent=False),
            "(n m) ... -> n m ...",
            m=minibatch_size,
        ),
        rollout_data,
    )

"""Shared utilities for Anakin agents."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from chex import Array

from dopamax.agents.anakin.base import AnakinAgent


def update_target_network(
    online_params: dict,
    target_params: dict,
    tau: float,
    train_step: int,
    update_interval: int,
    learning_starts: int,
) -> dict:
    """Update target network using Polyak averaging.

    Args:
        online_params: The online network parameters.
        target_params: The current target network parameters.
        tau: The soft update coefficient (0 < tau <= 1). tau=1 means hard update.
        train_step: The current training step.
        update_interval: Update target network every N steps.
        learning_starts: Only update after this many steps.

    Returns:
        Updated target network parameters.
    """
    should_update = (train_step > learning_starts) and (train_step % update_interval == 0)

    def update_fn(online, target):
        return jax.lax.select(should_update, tau * online + (1 - tau) * target, target)

    return jax.tree.map(update_fn, online_params, target_params)


def compute_importance_weights(
    priorities: Array,
    beta: float,
    beta_schedule_fn: Callable[[int], float] | None,
    train_step: int,
) -> Array:
    """Compute importance sampling weights for prioritized experience replay.

    Args:
        priorities: The priorities from the replay buffer.
        beta: The current beta value (or will be computed from schedule).
        beta_schedule_fn: Function that takes train_step and returns beta.
        train_step: The current training step.

    Returns:
        Normalized importance weights.
    """
    if beta_schedule_fn is not None:
        beta = beta_schedule_fn(train_step)

    importance_weights = 1.0 / priorities
    importance_weights **= beta
    importance_weights /= jnp.max(importance_weights)

    return importance_weights


def all_reduce_metrics(
    metrics: dict,
    agent: AnakinAgent,
) -> dict:
    """All-reduce metrics across devices/batches.

    Args:
        metrics: Dictionary of metrics to reduce.
        agent: The agent instance (for accessing all_reduce method).

    Returns:
        All-reduced metrics.
    """
    return agent._maybe_all_reduce("pmean", metrics)


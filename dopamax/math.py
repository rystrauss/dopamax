import jax
import jax.numpy as jnp
from chex import Array

from dopamax.agents.agent import Agent


def explained_variance(targets: Array, preds: Array, num_envs_per_device: int, num_devices: int) -> Array:
    """Computes the explained variance between predictions and targets.

    Values closer to 1.0 mean that the targets and predictions are highly correlated.

    Args:
        targets: The target values.
        preds: The predicted values.
        num_envs_per_device: The agent's number of environments per device.
        num_devices: The agent's number of devices.

    Returns:
        The scalar percentage of variance in targets that is explained by preds.
    """
    targets_2e = jnp.mean(targets**2)
    targets_e = jnp.mean(targets)

    diff = targets - preds
    diff_2e = jnp.mean(diff**2)
    diff_e = jnp.mean(diff)

    if num_envs_per_device > 1:
        targets_2e = jax.lax.pmean(targets_2e, Agent.batch_axis)
        targets_e = jax.lax.pmean(targets_e, Agent.batch_axis)
        diff_2e = jax.lax.pmean(diff_2e, Agent.batch_axis)
        diff_e = jax.lax.pmean(diff_e, Agent.batch_axis)

    if num_devices > 1:
        targets_2e = jax.lax.pmean(targets_2e, Agent.device_axis)
        targets_e = jax.lax.pmean(targets_e, Agent.device_axis)
        diff_2e = jax.lax.pmean(diff_2e, Agent.device_axis)
        diff_e = jax.lax.pmean(diff_e, Agent.device_axis)

    targets_e2 = targets_e**2
    diff_e2 = diff_e**2

    targets_var = targets_2e - targets_e2
    diff_var = diff_2e - diff_e2

    return jnp.maximum(-1.0, 1 - (diff_var / targets_var))

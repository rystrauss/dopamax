import jax.numpy as jnp
from chex import Array


def explained_variance(targets: Array, preds: Array) -> Array:
    """Computes the explained variance between predictions and targets.

    Values closer to 1.0 mean that the targets and predictions are highly correlated.

    Args:
        targets: The target values.
        preds: The predicted values.

    Returns:
        The scalar percentage of variance in targets that is explained by preds.
    """
    y_var = jnp.var(targets, axis=0)
    diff_var = jnp.var(targets - preds, axis=0)
    return jnp.maximum(-1.0, 1 - (diff_var / y_var))

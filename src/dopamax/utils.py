import functools

import jax
import jax.numpy as jnp
from distrax import Distribution


def expand_apply(f):
    """Wraps f to temporarily add a size-1 axis to its inputs.

    This is a subtle modification of hk.expand_apply that makes it compatible with functions that return
    distrax.Distribution objects.

    Syntactic sugar for::

        ins = jax.tree_util.tree_map(lambda t: np.expand_dims(t, axis=0), ins)
        out = f(ins)
        out = jax.tree_util.tree_map(lambda t: np.squeeze(t, axis=0), out)

    This may be useful for applying a function built for ``[Time, Batch, ...]``
    arrays to a single timestep.

    Args:
      f: The callable to be applied to the expanded inputs.

    Returns:
      f, wrapped as described above.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        expand = lambda t: jnp.expand_dims(t, axis=0)
        args = jax.tree_util.tree_map(expand, args)
        kwargs = jax.tree_util.tree_map(expand, kwargs)
        outputs = f(*args, **kwargs)

        return jax.tree_util.tree_map(
            lambda t: t[0],
            outputs,
            is_leaf=lambda o: isinstance(o, Distribution)
            or jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(o)),
        )

    return wrapper

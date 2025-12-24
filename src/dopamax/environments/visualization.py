from io import BytesIO

import jax
from brax.io import image

from dopamax.environments.brax.base import BraxEnvironment, BraxEnvState
from dopamax.environments.environment import Environment, EnvState


def render_trajectory(env: Environment, trajectory: EnvState) -> tuple[BytesIO, str]:
    if isinstance(env, BraxEnvironment):
        if not isinstance(trajectory, BraxEnvState):
            msg = f"Expected BraxEnvState for BraxEnvironment, got {type(trajectory).__name__}"
            raise TypeError(msg)

        return _render_brax_trajectory(env, trajectory)

    msg = f"Rendering not implemented for the environment: {env.name}."
    raise NotImplementedError(msg)


def _render_brax_trajectory(env: BraxEnvironment, trajectory: BraxEnvState) -> tuple[BytesIO, str]:
    n = trajectory.time.shape[0]
    trajectory = [jax.tree.map(lambda x: x[i], trajectory.brax_state.pipeline_state) for i in range(n)]
    bytes = image.render(env._brax_env.sys, trajectory, fmt="gif")
    return BytesIO(bytes), "gif"

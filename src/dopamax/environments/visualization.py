from io import BytesIO
from typing import Tuple

import jax
from brax.io import image

from dopamax.environments.brax.base import BraxEnvironment, BraxEnvState
from dopamax.environments.environment import EnvState, Environment


def render_trajectory(env: Environment, trajectory: EnvState) -> Tuple[BytesIO, str]:
    if isinstance(env, BraxEnvironment):
        assert isinstance(trajectory, BraxEnvState)

        return _render_brax_trajectory(env, trajectory)

    raise NotImplementedError(f"Rendering not implemented for the environment: {env.name}.")


def _render_brax_trajectory(env: BraxEnvironment, trajectory: BraxEnvState) -> Tuple[BytesIO, str]:
    n = trajectory.time.shape[0]
    trajectory = [jax.tree.map(lambda x: x[i], trajectory.brax_state.pipeline_state) for i in range(n)]
    bytes = image.render(env._brax_env.sys, trajectory, fmt="gif")
    return BytesIO(bytes), "gif"

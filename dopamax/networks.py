from typing import Sequence, Optional, Callable, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

from dopamax.spaces import Space, Discrete, Box
from dopamax.typing import Observation

_registry = {}

# Function that takes no arguments and returns a network.
NetworkBuildFn = Callable[[], hk.Module]


def register(name: str):
    """Decorator that registers a network build function."""

    def _fn(cls):
        _registry[name] = cls
        return cls

    return _fn


def get_network_build_fn(name: str, **kwargs) -> NetworkBuildFn:
    """Gets a network build function by name.

    Args:
        name: The name of the network build function to get.
        kwargs: Keyword arguments to pass to the network build function constructor.

    Returns:
        The network build function.
    """
    return _registry[name](**kwargs)


@register("mlp")
def get_mlp_build_fn(hidden_units: Sequence[int] = (64, 64), activation: str = "relu") -> NetworkBuildFn:
    """Gets a network build function for a multi-layer perceptron.

    Args:
        hidden_units: A sequence of integers representing the number of hidden units in each layer.
        activation: The activation function to use in each layer. Must exist in `jax.nn`.

    Returns:
        A network build function for the specified MLP.
    """
    return lambda: hk.Sequential(
        [
            hk.Flatten(),
            hk.nets.MLP(
                hidden_units,
                activation=getattr(jax.nn, activation),
                activate_final=True,
                w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0)),
            ),
        ],
    )


def get_actor_critic_model_fn(
    action_space: Space,
    network_build_fn: NetworkBuildFn,
    value_network: Optional[str] = None,
    free_log_std: bool = False,
) -> Callable[[Observation], distrax.Distribution | Tuple[distrax.Distribution, Array]]:
    """Gets a model function for a generic actor-critic agent.

    Here, a model refers to a function that takes observations as input and returns a policy distribution and a value
    estimate.

    Args:
        action_space: The action space that the policy should operate in.
        network_build_fn: A network build function that will be used to construct the model's networks.
        value_network: The type of value network to use. If `None`, the model will not have a value estimate and will
            only output a policy distribution. If "copy", the value network be a copy of the policy network, but with
            independent parameters. If "shared", the value network will share the same parameters as the policy network
            (excluding the output layer).
        free_log_std: If True, the scales of Gaussian policies will be learned as free parameters. Otherwise, they will
            be functions of the observations.

    Returns:
        A model function for the specified actor-critic agent.
    """

    def model_fn(observations: Observation) -> distrax.Distribution | Tuple[distrax.Distribution, Array]:
        policy_net = network_build_fn()

        policy_net_output = policy_net(observations)

        if isinstance(action_space, Discrete):
            num_policy_outputs = action_space.n
            policy_params = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(policy_net_output)
            policy = distrax.Categorical(policy_params)
        elif isinstance(action_space, Box):
            if free_log_std:
                num_policy_outputs = action_space.shape[0]
                log_std = hk.get_parameter("log_std", (num_policy_outputs,), init=hk.initializers.Constant(0.0))
                loc = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(policy_net_output)
            else:
                num_policy_outputs = action_space.shape[0] * 2
                policy_params = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(
                    policy_net_output
                )
                loc, log_std = jnp.split(policy_params, 2, axis=-1)

            policy = distrax.MultivariateNormalDiag(loc, jnp.exp(log_std))
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

        if value_network is None:
            return policy

        if value_network == "shared":
            values = hk.Linear(1, w_init=hk.initializers.Orthogonal(1.0))(policy_net_output)
        elif value_network == "copy":
            value_net = network_build_fn()
            values = hk.Linear(1, w_init=hk.initializers.Orthogonal(1.0))(value_net(observations))
        else:
            raise ValueError(f"Unsupported value network: {value_network}")

        values = jnp.squeeze(values, axis=-1)
        return policy, values

    return model_fn


def get_discrete_q_network_model_fn(
    action_space: Discrete,
    network_build_fn: NetworkBuildFn,
    final_hidden_units: Sequence[int] = tuple(),
    dueling: bool = False,
) -> Callable[[Observation], Array]:
    def model_fn(observations: Observation) -> Array:
        base_net = network_build_fn()
        base_net_output = base_net(observations)

        if dueling:
            assert (
                final_hidden_units
            ), "Dueling networks must have at least one hidden layer provided in `final_hidden_units`."

            advantage_net = hk.Sequential(
                [
                    hk.nets.MLP(final_hidden_units, w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0))),
                    hk.Linear(action_space.n, w_init=hk.initializers.Orthogonal(0.01)),
                ]
            )
            value_net = hk.Sequential(
                [
                    hk.nets.MLP(final_hidden_units, w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0))),
                    hk.Linear(1, w_init=hk.initializers.Orthogonal(0.01)),
                ]
            )

            advantage_stream = advantage_net(base_net_output)
            value_stream = value_net(base_net_output)

            output = value_stream + (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))
        else:
            output_net = hk.Sequential(
                [
                    hk.nets.MLP(final_hidden_units, w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0))),
                    hk.Linear(action_space.n, w_init=hk.initializers.Orthogonal(0.01)),
                ]
            )

            output = output_net(base_net_output)

        return output

    return model_fn

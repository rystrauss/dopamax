from typing import Sequence, Optional, Callable, Tuple, Union

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

from src.dopamax import Transformed
from src.dopamax.spaces import Space, Discrete, Box
from src.dopamax.typing import Observation, Action

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


class CNN(hk.Module):
    """A basic convolutional neural network module.

    Args:
        conv_layers: A sequence of tuples representing the number of channels, kernel size, and stride for each layer.
        activation: The activation function to use in each layer. Must exist in `jax.nn`.
        activate_final: Whether to apply the activation function to the final outputs.
    """

    def __init__(
        self,
        conv_layers: Sequence[Tuple[int, int, int]],
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        activate_final: bool = False,
    ):
        super().__init__()
        self._conv_layers = conv_layers
        self._activation = activation
        self._activate_final = activate_final

    def __call__(self, x: Array) -> Array:
        for i, (channels, kernel, stride) in enumerate(self._conv_layers):
            x = hk.Conv2D(channels, kernel, stride, padding="SAME")(x)
            if i < len(self._conv_layers) - 1 or self._activate_final:
                x = self._activation(x)

        return x


NATURE_CNN_CONV_LAYERS = ((32, 8, 4), (64, 4, 2), (64, 3, 1))


@register("cnn")
def get_cnn_build_fn(
    conv_layers: Sequence[Tuple[int, int, int]] = NATURE_CNN_CONV_LAYERS,
    hidden_units: Sequence[int] = (512,),
    activation: str = "relu",
) -> NetworkBuildFn:
    """Gets a network build function for a CNN.

    Default values for keyword arguments are for the CNN used in the DQN Nature paper.

    Args:
        conv_layers: A sequence of tuples representing the number of channels, kernel size, and stride for each layer.
        hidden_units: A sequence of integers representing the number of hidden units in each fully-connected layer
            after the convolutions.
        activation: The activation function to use in each layer. Must exist in `jax.nn`.

    Returns:
        A network build function for the specified MLP.
    """
    return lambda: hk.Sequential(
        [
            CNN(conv_layers, activation=getattr(jax.nn, activation), activate_final=True),
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
    tanh_value: bool = False,
    tanh_policy: bool = False,
) -> Callable[[Observation], Union[distrax.Distribution, Tuple[distrax.Distribution, Array]]]:
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
        tanh_value: If True, the value estimate will be passed through a tanh activation function.
        tanh_policy: Whether to apply a Tanh bijector to the policy (only works for continuous action spaces).

    Returns:
        A model function for the specified actor-critic agent.
    """

    def model_fn(observations: Observation) -> Union[distrax.Distribution, Tuple[distrax.Distribution, Array]]:
        policy_net = network_build_fn()

        policy_net_output = policy_net(observations)

        if isinstance(action_space, Discrete):
            num_policy_outputs = action_space.n
            policy_params = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(policy_net_output)
            policy = distrax.Categorical(policy_params)
        elif isinstance(action_space, Box):
            if free_log_std:
                num_policy_outputs = action_space.shape[0]
                loc = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(policy_net_output)
                log_std = hk.get_parameter("log_std", (num_policy_outputs,), init=hk.initializers.Constant(0.0))
                log_std = jnp.broadcast_to(log_std, loc.shape)
            else:
                num_policy_outputs = action_space.shape[0] * 2
                policy_params = hk.Linear(num_policy_outputs, w_init=hk.initializers.Orthogonal(0.01))(
                    policy_net_output
                )
                loc, log_std = jnp.split(policy_params, 2, axis=-1)

            log_std = jnp.clip(log_std, -20, 2)
            policy = distrax.MultivariateNormalDiag(loc, jnp.exp(log_std))

            if tanh_policy:
                policy = Transformed(policy, distrax.Block(distrax.Tanh(), ndims=1))

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

        if tanh_value:
            values = jax.nn.tanh(values)

        return policy, values

    return model_fn


def get_deterministic_actor_model_fn(
    action_space: Box,
    network_build_fn: NetworkBuildFn,
    final_hidden_units: Sequence[int] = tuple(),
) -> Callable[[Observation], Array]:
    bounded = bool(jnp.logical_and(action_space.bounded_above, action_space.bounded_below).any())

    def model_fn(observations: Observation) -> Array:
        base_net = network_build_fn()
        base_net_output = base_net(observations)

        policy_net = hk.Sequential(
            [
                hk.nets.MLP(final_hidden_units, w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0))),
                hk.Linear(action_space.shape[0], w_init=hk.initializers.Orthogonal(0.01)),
            ]
        )
        policy_output = policy_net(base_net_output)

        action_space_low = action_space.low[jnp.newaxis]
        action_space_high = action_space.high[jnp.newaxis]

        if bounded:

            def squash(x):
                x = jax.nn.sigmoid(2 * x)
                return (action_space_high - action_space_low) * x + action_space_low

            policy_output = squash(policy_output)

        return policy_output

    return model_fn


def get_discrete_q_network_model_fn(
    action_space: Discrete,
    network_build_fn: NetworkBuildFn,
    final_hidden_units: Sequence[int] = tuple(),
    dueling: bool = False,
    use_twin: bool = False,
) -> Union[Callable[[Observation], Array], Callable[[Observation], Tuple[Array, Array]]]:
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

    if use_twin:

        def twin_model_fn(observations: Observation) -> Tuple[Array, Array]:
            q1_fn = get_discrete_q_network_model_fn(action_space, network_build_fn, final_hidden_units)
            q2_fn = get_discrete_q_network_model_fn(action_space, network_build_fn, final_hidden_units)

            return q1_fn(observations), q2_fn(observations)

        return twin_model_fn

    return model_fn


def get_continuous_q_network_model_fn(
    observation_space: Space,
    network_build_fn: NetworkBuildFn,
    final_hidden_units: Sequence[int] = tuple(),
    use_twin: bool = False,
) -> Union[Callable[[Observation, Action], Array], Callable[[Observation, Action], Tuple[Array, Array]]]:
    def model_fn(observations: Observation, actions: Action) -> Array:
        base_net = network_build_fn()

        output_net = hk.Sequential(
            [
                hk.nets.MLP(final_hidden_units, w_init=hk.initializers.Orthogonal(jnp.sqrt(2.0))),
                hk.Linear(1, w_init=hk.initializers.Orthogonal(0.01)),
            ]
        )

        concat_at_front = len(observation_space.shape) == 1

        if concat_at_front:
            latent = base_net(jnp.concatenate([observations, actions], axis=-1))
        else:
            latent = base_net(observations)
            latent = jnp.concatenate([latent, actions])

        return jnp.squeeze(output_net(latent), axis=-1)

    if use_twin:

        def twin_model_fn(observations: Observation, actions: Action) -> Tuple[Array, Array]:
            q1_fn = get_continuous_q_network_model_fn(observation_space, network_build_fn, final_hidden_units)
            q2_fn = get_continuous_q_network_model_fn(observation_space, network_build_fn, final_hidden_units)

            return q1_fn(observations, actions), q2_fn(observations, actions)

        return twin_model_fn

    return model_fn

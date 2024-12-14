from functools import partial
from typing import Tuple, Dict

import distrax
import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import PRNGKey, ArrayTree
from ml_collections import ConfigDict

from dopamax.agents.anakin.base import AnakinAgent, AnakinTrainStateWithReplayBuffer, AnakinTrainState
from dopamax.agents.utils import register
from dopamax.environments.environment import Environment
from dopamax.networks import get_network_build_fn, get_discrete_q_network_model_fn
from dopamax.prioritized_item_buffer import create_prioritised_item_buffer
from dopamax.rollouts import rollout_truncated, SampleBatch
from dopamax.spaces import Discrete
from dopamax.typing import Metrics, Observation, Action
from dopamax.utils import expand_apply

_DEFAULT_DQN_CONFIG = ConfigDict(
    {
        # The initial epsilon value for epsilon-greedy exploration.
        "initial_epsilon": 1.0,
        # The final epsilon value for epsilon-greedy exploration.
        "final_epsilon": 0.02,
        # The number of training steps over which to anneal epsilon.
        "epsilon_decay_steps": 10000,
        # The discount factor.
        "gamma": 0.99,
        # The learning rate.
        "learning_rate": 5e-4,
        # The number of training steps to take before beginning gradient updates.
        "learning_starts": 1000,
        # The size of experience batches to sample from the replay buffer and learn on.
        "batch_size": 32,
        # The maximum size of the replay buffer.
        "buffer_size": 10000,
        # The frequency with which the target network is updated.
        "target_update_freq": 500,
        # The type of network to use.
        "network": "mlp",
        # The configuration dictionary for the network.
        "network_config": {"hidden_units": [64, 64]},
        # Additional fully-connected layers to add after the base network before the output layer. At least one layer
        # must be added when using the dueling architecture.
        "final_hidden_units": (),
        # Whether to use double q-learning.
        "double": True,
        # Whether to use a dueling q-network architecture.
        "dueling": False,
        # Whether to use a prioritized experience replay.
        "prioritized_replay": False,
        # The alpha parameter used for prioritized experience replay.
        "prioritized_replay_alpha": 0.6,
        # The initial beta parameter used for prioritized experience replay.
        "initial_prioritized_replay_beta": 0.4,
        # The final beta parameter used for prioritized experience replay.
        "final_prioritized_replay_beta": 1.0,
        # The number of steps over which to decay the beta parameter used for prioritized experience replay.
        "prioritized_replay_beta_decay_steps": 1,
    }
)


@register("DQN")
class DQN(AnakinAgent):
    """Deep Q-Network (DQN) agent.

    This is a simple DQN implementation that is very close to the original paper. It doesn't have a lot of the more
    recent bells and whistles that have been developed.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.

    References:
        https://www.nature.com/articles/nature14236
    """

    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        assert isinstance(self.env.action_space, Discrete), "DQN only supports discrete action spaces."

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        model_fn = get_discrete_q_network_model_fn(
            env.action_space, network_build_fn, self.config.final_hidden_units, self.config.dueling
        )
        self._model = hk.transform(model_fn)

        def policy_fn(
            params: hk.Params, key: PRNGKey, observation: Observation, epsilon: float
        ) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, sample_key = jax.random.split(key)

            preferences = expand_apply(partial(self._model.apply, params, model_key))(observation)
            action = distrax.EpsilonGreedy(preferences, epsilon).sample(seed=sample_key)

            return action, {}

        self._epsilon_schedule = optax.linear_schedule(
            self.config.initial_epsilon, self.config.final_epsilon, self.config.epsilon_decay_steps
        )
        self._optimizer = optax.adam(self.config.learning_rate)

        self._rollout_fn = partial(rollout_truncated, env, 1, policy_fn)

        temp_key = jax.random.PRNGKey(0)
        temp_train_state = self._initial_train_state_without_replay_buffer(temp_key, temp_key)
        rollout_data_spec, _, _, _ = jax.eval_shape(
            self._rollout_fn,
            temp_train_state.params["online"],
            temp_train_state.key,
            temp_train_state.time_step,
            temp_train_state.env_state,
            epsilon=1.0,
        )
        rollout_data = jax.tree.map(lambda x: jnp.empty(x.shape, x.dtype), rollout_data_spec)

        if self.config.prioritized_replay:
            self._buffer = create_prioritised_item_buffer(
                max_length=self.config.buffer_size,
                min_length=self.config.batch_size,
                sample_batch_size=self.config.batch_size,
                add_batches=False,
                add_sequences=False,
                priority_exponent=self.config.prioritized_replay_alpha,
                device=jax.default_backend(),
            )

            self._beta_schedule = optax.linear_schedule(
                self.config.initial_prioritized_replay_beta,
                self.config.final_prioritized_replay_beta,
                self.config.prioritized_replay_beta_decay_steps,
            )
        else:
            self._buffer = fbx.make_item_buffer(
                max_length=self.config.buffer_size,
                min_length=self.config.batch_size,
                sample_batch_size=self.config.batch_size,
            )

            self._beta_schedule = lambda _: 1.0
        self._buffer_initial_state = self._buffer.init(rollout_data)

    @staticmethod
    def default_config() -> ConfigDict:
        config = super(DQN, DQN).default_config()
        config.update(_DEFAULT_DQN_CONFIG)
        return config

    def compute_action(
        self,
        params: hk.Params,
        key: PRNGKey,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        preferences = self._model.apply(params["online"], key, observation)
        pi = distrax.Categorical(logits=preferences)
        return pi.mode() if deterministic else pi.sample(seed=key)

    def _initial_train_state_without_replay_buffer(
        self, consistent_key: PRNGKey, divergent_key: PRNGKey
    ) -> AnakinTrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key), 0)
        params = self._model.init(consistent_key, sample_obs)
        params = {"online": params, "target": params}
        time_step, env_state = self.env.reset(env_key)
        opt_state = self._optimizer.init(params["online"])

        return AnakinTrainState.initial(train_state_key, params, opt_state, time_step, env_state)

    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> AnakinTrainStateWithReplayBuffer:
        train_state_without_replay_buffer = self._initial_train_state_without_replay_buffer(
            consistent_key, divergent_key
        )
        return AnakinTrainStateWithReplayBuffer(
            **dict(train_state_without_replay_buffer), buffer_state=self._buffer_initial_state
        )

    def _loss(self, online_params, target_params, key, obs, actions, rewards, next_obs, discounts, weights):
        next_q_values = self._model.apply(target_params, key, next_obs)
        q_values = self._model.apply(online_params, key, obs)

        if self.config.double:
            next_q_selectors = self._model.apply(online_params, key, next_obs)
            td_error = jax.vmap(rlax.double_q_learning)(
                q_values, actions, rewards, discounts, next_q_values, next_q_selectors
            )
        else:
            td_error = jax.vmap(rlax.q_learning)(q_values, actions, rewards, discounts, next_q_values)

        loss = jnp.mean(rlax.l2_loss(td_error * weights))

        metrics = {
            "loss": loss,
            "td_errors": td_error,
        }

        return loss, metrics

    def _update(self, train_step, params, opt_state, key, obs, actions, rewards, next_obs, discounts, weights):
        grads, info = jax.grad(self._loss, has_aux=True)(
            params["online"], params["target"], key, obs, actions, rewards, next_obs, discounts, weights
        )

        td_errors = info.pop("td_errors")

        grads, info = self._maybe_all_reduce("pmean", (grads, info))

        updates, new_opt_state = self._optimizer.update(grads, opt_state, params["online"])
        new_online_params = optax.apply_updates(params["online"], updates)

        new_target_params = jax.tree.map(
            lambda a, b: jax.lax.select(
                jnp.logical_and(
                    jnp.any(train_step > self.config.learning_starts),
                    jnp.any(train_step % self.config.target_update_freq == 0),
                ),
                a,
                b,
            ),
            new_online_params,
            params["target"],
        )

        new_params = {"online": new_online_params, "target": new_target_params}

        return new_params, new_opt_state, info, td_errors

    def train_step(
        self, train_state: AnakinTrainStateWithReplayBuffer
    ) -> Tuple[AnakinTrainStateWithReplayBuffer, Metrics]:
        next_train_state_key, rollout_key, update_key, sample_key = jax.random.split(train_state.key, 4)

        current_epsilon = self._epsilon_schedule(train_state.train_step)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params["online"],
            rollout_key,
            train_state.time_step,
            train_state.env_state,
            epsilon=current_epsilon,
        )

        new_buffer_state = self._buffer.add(train_state.buffer_state, rollout_data)

        batch = self._buffer.sample(new_buffer_state, sample_key)
        sample = batch.experience
        priorities = batch.priorities if self.config.prioritized_replay else 1.0

        importance_weights = 1.0 / priorities
        importance_weights **= self._beta_schedule(train_state.train_step)
        importance_weights /= jnp.max(importance_weights)

        sample = jax.tree.map(lambda x: jnp.squeeze(x, 1), sample)

        new_params, new_opt_state, metrics, td_error = self._update(
            train_state.train_step,
            train_state.params,
            train_state.opt_state,
            update_key,
            sample[SampleBatch.OBSERVATION],
            sample[SampleBatch.ACTION],
            sample[SampleBatch.REWARD],
            sample[SampleBatch.NEXT_OBSERVATION],
            sample[SampleBatch.DISCOUNT],
            importance_weights,
        )

        if self.config.prioritized_replay:
            new_priorities = jnp.abs(td_error) + 1e-6
            new_buffer_state = self._buffer.set_priorities(new_buffer_state, batch.indices, new_priorities)

        metrics = jax.tree.map(jnp.mean, metrics)

        next_train_state = train_state.update(
            new_key=next_train_state_key,
            rollout_data=rollout_data,
            new_params=new_params,
            new_opt_state=new_opt_state,
            new_time_step=new_time_step,
            new_env_state=new_env_state,
            new_buffer_state=new_buffer_state,
            maybe_all_reduce_fn=self._maybe_all_reduce,
        )

        warmup_train_state = train_state.update(
            new_key=next_train_state_key,
            rollout_data=rollout_data,
            new_params=train_state.params,
            new_opt_state=train_state.opt_state,
            new_time_step=new_time_step,
            new_env_state=new_env_state,
            new_buffer_state=new_buffer_state,
            maybe_all_reduce_fn=self._maybe_all_reduce,
        )

        next_train_state = jax.tree.map(
            lambda a, b: jax.lax.select(jnp.any(train_state.train_step > self.config.learning_starts), a, b),
            next_train_state,
            warmup_train_state,
        )

        return next_train_state, metrics

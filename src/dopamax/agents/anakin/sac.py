import math
from functools import partial
from typing import Tuple, Dict

import flashbax as fbx
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from chex import PRNGKey, ArrayTree
from ml_collections import ConfigDict

from dopamax.agents.anakin.base import AnakinAgent, AnakinTrainState, AnakinTrainStateWithReplayBuffer
from dopamax.agents.utils import register
from dopamax.environments.environment import Environment
from dopamax.networks import (
    get_network_build_fn,
    get_actor_critic_model_fn,
    get_discrete_q_network_model_fn,
    get_continuous_q_network_model_fn,
)
from dopamax.prioritized_item_buffer import create_prioritised_item_buffer
from dopamax.rollouts import rollout_truncated, SampleBatch
from dopamax.spaces import Discrete
from dopamax.typing import Metrics, Observation, Action
from dopamax.utils import expand_apply

_DEFAULT_SAC_CONFIG = ConfigDict(
    {
        "actor_learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "entropy_learning_rate": 3e-4,
        "learning_starts": 1500,
        "tau": 5e-3,
        "target_update_interval": 1,
        "initial_alpha": 1.0,
        "target_entropy": "auto",
        "gamma": 0.95,
        "buffer_size": 50000,
        "train_freq": 1,
        "batch_size": 256,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "initial_prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_decay_steps": 1,
        "network": "mlp",
        "network_config": {"hidden_units": [64, 64]},
        "final_hidden_units": (),
    }
)


@register("SAC")
class SAC(AnakinAgent):
    """The Soft Actor-Critic algorithm.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.

    References:
        https://arxiv.org/abs/1812.05905
    """

    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        self._discrete = isinstance(env.action_space, Discrete)

        if self.config.target_entropy == "auto":
            if self._discrete:
                self._target_entropy = 0.98 * jnp.array(-jnp.log(1.0 / env.action_space.n), dtype=jnp.float32)
            else:
                self._target_entropy = -math.prod(env.action_space.shape)
        else:
            self._target_entropy = self.config.target_entropy

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)

        actor_fn = get_actor_critic_model_fn(
            env.action_space,
            network_build_fn,
            value_network=None,
            tanh_policy=True,
        )

        q_fn = (
            get_discrete_q_network_model_fn(
                env.action_space, network_build_fn, self.config.final_hidden_units, use_twin=True
            )
            if self._discrete
            else get_continuous_q_network_model_fn(
                env.observation_space,
                network_build_fn,
                self.config.final_hidden_units,
                use_twin=True,
            )
        )

        self._actor = hk.transform(actor_fn)
        self._critic = hk.transform(q_fn)

        self._action_space_low = env.action_space.low if not self._discrete else None
        self._action_space_high = env.action_space.high if not self._discrete else None

        def policy_fn(
            actor_params: hk.Params,
            key: PRNGKey,
            observation: Observation,
            uniform_sample: bool,
        ) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, sample_key = jax.random.split(key)

            pi = expand_apply(partial(self._actor.apply, actor_params, model_key))(observation)

            actions = jax.lax.select(
                uniform_sample,
                self.action_space.sample(sample_key),
                pi.sample(seed=sample_key),
            )

            log_probs = pi.log_prob(actions)

            if not self._discrete:
                actions = ((actions + 1.0) / 2.0 + self._action_space_low) * (
                    self._action_space_high - self._action_space_low
                )

            return actions, {SampleBatch.ACTION_LOGP: log_probs}

        self._actor_optimizer = optax.adam(self.config.actor_learning_rate)
        self._critic_optimizer = optax.adam(self.config.critic_learning_rate)
        self._alpha_optimizer = optax.adam(self.config.entropy_learning_rate)

        self._rollout_fn = partial(rollout_truncated, env, 1, policy_fn)

        temp_key = jax.random.PRNGKey(0)
        temp_train_state = self._initial_train_state_without_replay_buffer(temp_key, temp_key)

        rollout_data_spec, _, _, _ = jax.eval_shape(
            self._rollout_fn,
            temp_train_state.params["online"]["actor"],
            temp_train_state.key,
            temp_train_state.time_step,
            temp_train_state.env_state,
            uniform_sample=False,
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
        config = super(SAC, SAC).default_config()
        config.update(_DEFAULT_SAC_CONFIG)
        return config

    def compute_action(
        self,
        params: hk.Params,
        key: PRNGKey,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        sample_key, model_key = jax.random.split(key)

        pi = self._actor.apply(params["online"], model_key, observation)

        if not deterministic:
            action = pi.mode() if self._discrete else pi.mean()
        else:
            action = pi.sample(sample_key)

        return action

    def _initial_train_state_without_replay_buffer(
        self, consistent_key: PRNGKey, divergent_key: PRNGKey
    ) -> AnakinTrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key), 0)
        sample_action = jnp.expand_dims(self.env.action_space.sample(consistent_key), 0)
        actor_params = self._actor.init(consistent_key, sample_obs)
        if self._discrete:
            critic_params = self._critic.init(consistent_key, sample_obs)
        else:
            critic_params = self._critic.init(consistent_key, sample_obs, sample_action)
        params = {
            "online": {"actor": actor_params, "critic": critic_params},
            "target": {"critic": critic_params},
            "log_alpha": jnp.log(self.config.initial_alpha),
        }
        time_step, env_state = self.env.reset(env_key)
        actor_opt_state = self._actor_optimizer.init(params["online"]["actor"])
        critic_opt_state = self._critic_optimizer.init(params["online"]["critic"])
        alpha_opt_state = self._alpha_optimizer.init(params["log_alpha"])
        opt_state = {"actor": actor_opt_state, "critic": critic_opt_state, "log_alpha": alpha_opt_state}

        return AnakinTrainState.initial(train_state_key, params, opt_state, time_step, env_state)

    def initial_train_state(self, consistent_key: PRNGKey, divergent_key: PRNGKey) -> AnakinTrainStateWithReplayBuffer:
        train_state_without_replay_buffer = self._initial_train_state_without_replay_buffer(
            consistent_key, divergent_key
        )
        return AnakinTrainStateWithReplayBuffer(
            **dict(train_state_without_replay_buffer), buffer_state=self._buffer_initial_state
        )

    def _critic_loss(
        self,
        online_critic_params,
        target_critic_params,
        online_actor_params,
        log_alpha,
        key,
        obs,
        actions,
        rewards,
        next_obs,
        discounts,
        weights,
    ):
        actor_key, critic_key, target_critic_key, sample_key = jax.random.split(key, 4)

        alpha = jnp.exp(log_alpha)

        next_pi = self._actor.apply(online_actor_params, actor_key, next_obs)

        if self._discrete:
            next_logpacs = jax.nn.log_softmax(next_pi.logits)
            q_targets = jnp.minimum(*self._critic.apply(target_critic_params, target_critic_key, next_obs))
            q_targets = jnp.sum(jnp.exp(next_logpacs) * (q_targets - alpha * next_logpacs), axis=-1)
            backup = jnp.expand_dims(rewards + self.config.gamma * discounts * q_targets, axis=1)
            q1_values, q2_values = self._critic.apply(online_critic_params, critic_key, obs)
        else:
            next_actions, next_logpacs = next_pi.sample_and_log_prob(seed=sample_key)
            q_targets = jnp.minimum(
                *self._critic.apply(target_critic_params, target_critic_key, next_obs, next_actions)
            )
            backup = rewards + self.config.gamma * discounts * (q_targets - alpha * next_logpacs)
            q1_values, q2_values = self._critic.apply(online_critic_params, critic_key, obs, actions)

        q1_loss = rlax.huber_loss(q1_values - backup)
        q2_loss = rlax.huber_loss(q2_values - backup)
        critic_loss = 0.5 * (q1_loss + q2_loss)
        td_error = 0.5 * ((q1_values - backup) + (q2_values - backup))

        critic_loss = jnp.mean(critic_loss * weights)

        return critic_loss, {"critic_loss": critic_loss, "td_error": td_error}

    def _actor_loss(self, online_actor_params, online_critic_params, log_alpha, key, obs):
        actor_key, critic_key, sample_key = jax.random.split(key, 3)

        alpha = jnp.exp(log_alpha)

        pi = self._actor.apply(online_actor_params, actor_key, obs)

        if self._discrete:
            log_pacs = jax.nn.log_softmax(pi.logits)
            q_targets = jnp.minimum(*self._critic.apply(online_critic_params, critic_key, obs))
            actor_loss = jnp.mean(
                jnp.sum(
                    jnp.exp(log_pacs) * (alpha * log_pacs - q_targets),
                    axis=-1,
                )
            )
        else:
            actions, log_pacs = pi.sample_and_log_prob(seed=sample_key)
            q_targets = jnp.minimum(*self._critic.apply(online_critic_params, critic_key, obs, actions))
            actor_loss = jnp.mean(alpha * log_pacs - q_targets)

        return actor_loss, {"actor_loss": actor_loss, "log_pacs": log_pacs}

    def _alpha_loss(self, log_alpha, log_pacs):
        loss = jnp.mean(jnp.sum(jnp.exp(log_pacs) * (-log_alpha * log_pacs + self._target_entropy), axis=-1))
        return loss, {"alpha_loss": loss}

    def _update(self, train_step, params, opt_state, key, obs, actions, rewards, next_obs, discounts, weights):
        critic_grads, critic_info = jax.grad(self._critic_loss, has_aux=True)(
            params["online"]["critic"],
            params["target"]["critic"],
            params["online"]["actor"],
            params["log_alpha"],
            key,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            weights,
        )

        actor_grads, actor_info = jax.grad(self._actor_loss, has_aux=True)(
            params["online"]["actor"], params["online"]["critic"], params["log_alpha"], key, obs
        )

        td_error = critic_info.pop("td_error")
        log_pacs = actor_info.pop("log_pacs")

        alpha_grads, alpha_info = jax.grad(self._alpha_loss, has_aux=True)(params["log_alpha"], log_pacs)

        info = {
            "actor_loss": actor_info["actor_loss"],
            "critic_loss": critic_info["critic_loss"],
            "alpha_loss": alpha_info["alpha_loss"],
        }

        critic_grads, actor_grads, alpha_grads, info = self._maybe_all_reduce(
            "pmean", (critic_grads, actor_grads, alpha_grads, info)
        )

        critic_updates, new_critic_opt_state = self._critic_optimizer.update(
            critic_grads, opt_state["critic"], params["online"]["critic"]
        )
        actor_updates, new_actor_opt_state = self._actor_optimizer.update(
            actor_grads, opt_state["actor"], params["online"]["actor"]
        )
        alpha_updates, new_alpha_opt_state = self._alpha_optimizer.update(
            alpha_grads, opt_state["log_alpha"], params["log_alpha"]
        )

        new_online_critic_params = optax.apply_updates(params["online"]["critic"], critic_updates)
        new_online_actor_params = optax.apply_updates(params["online"]["actor"], actor_updates)
        new_log_alpha = optax.apply_updates(params["log_alpha"], alpha_updates)

        new_online_params = {
            "critic": new_online_critic_params,
            "actor": new_online_actor_params,
        }

        new_target_params = jax.tree.map(
            lambda a, b: jax.lax.select(
                jnp.logical_and(
                    jnp.any(train_step > self.config.learning_starts),
                    jnp.any(train_step % self.config.target_update_interval == 0),
                ),
                # Polyak Update
                self.config.tau * a + (1 - self.config.tau) * b,
                b,
            ),
            {"critic": new_online_critic_params},
            params["target"],
        )

        new_params = {"online": new_online_params, "target": new_target_params, "log_alpha": new_log_alpha}
        new_opt_state = {"actor": new_actor_opt_state, "critic": new_critic_opt_state, "log_alpha": new_alpha_opt_state}

        return new_params, new_opt_state, info, td_error

    def train_step(
        self, train_state: AnakinTrainStateWithReplayBuffer
    ) -> Tuple[AnakinTrainStateWithReplayBuffer, Metrics]:
        next_train_state_key, rollout_key, update_key, sample_key = jax.random.split(train_state.key, 4)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params["online"]["actor"],
            rollout_key,
            train_state.time_step,
            train_state.env_state,
            uniform_sample=train_state.train_step <= self.config.learning_starts,
        )

        new_buffer_state = self._buffer.add(train_state.buffer_state, rollout_data)

        batch = self._buffer.sample(new_buffer_state, sample_key)
        sample = batch.experience
        priorities = batch.priorities if self.config.prioritized_replay else 1.0

        importance_weights = 1.0 / priorities
        importance_weights **= self._beta_schedule(train_state.train_step)
        importance_weights /= jnp.max(importance_weights)

        sample = jax.tree_map(lambda x: jnp.squeeze(x, 1), sample)

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

        metrics = jax.tree_map(jnp.mean, metrics)

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

        next_train_state = jax.tree_map(
            lambda a, b: jax.lax.select(jnp.any(train_state.train_step > self.config.learning_starts), a, b),
            next_train_state,
            warmup_train_state,
        )

        return next_train_state, metrics

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

from src.dopamax.agents import AnakinAgent, AnakinTrainStateWithReplayBuffer, AnakinTrainState
from src.dopamax.agents.utils import register
from src.dopamax import Environment
from src.dopamax import get_network_build_fn, get_deterministic_actor_model_fn, get_continuous_q_network_model_fn
from src.dopamax import create_prioritised_item_buffer
from src.dopamax import rollout_truncated, SampleBatch
from src.dopamax.spaces import Box
from src.dopamax.typing import Metrics, Observation, Action
from src.dopamax.utils import expand_apply

_DEFAULT_DDPG_CONFIG = ConfigDict(
    {
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "tau": 0.002,
        "gamma": 0.95,
        "buffer_size": 50000,
        "train_freq": 1,
        "target_update_interval": 1,
        "learning_starts": 1500,
        "random_steps": 1500,
        "batch_size": 256,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "initial_prioritized_replay_beta": 0.4,
        "final_prioritized_replay_beta": 1.0,
        "prioritized_replay_beta_decay_steps": 1,
        "initial_noise_scale": 0.1,
        "final_noise_scale": 0.1,
        "noise_scale_steps": 1,
        "use_huber": False,
        "use_twin_critic": False,
        "policy_delay": 1,
        "smooth_target_policy": False,
        "target_noise": 0.2,
        "target_noise_clip": 0.5,
        # The type of network to use.
        "network": "mlp",
        # The configuration dictionary for the network.
        "network_config": {"hidden_units": [64, 64]},
        # Additional fully-connected layers to add after the base network before the output layer.
        "final_hidden_units": (),
    }
)


@register("DDPG")
class DDPG(AnakinAgent):
    """Deep Deterministic Policy Gradients (DDPG) agent.

    Args:
        env: The environment to interact with.
        config: The configuration dictionary for the agent.

    References:
        https://arxiv.org/abs/1509.02971
    """

    def __init__(self, env: Environment, config: ConfigDict):
        super().__init__(env, config)

        assert isinstance(self.env.action_space, Box), "DDPG only supports continuous action spaces."

        network_build_fn = get_network_build_fn(self.config.network, **self.config.network_config)
        actor_fn = get_deterministic_actor_model_fn(
            env.action_space,
            network_build_fn,
            self.config.final_hidden_units,
        )
        q_fn = get_continuous_q_network_model_fn(
            env.observation_space, network_build_fn, self.config.final_hidden_units, self.config.use_twin_critic
        )
        self._actor = hk.transform(actor_fn)
        self._critic = hk.transform(q_fn)

        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high

        def policy_fn(
            actor_params: hk.Params,
            key: PRNGKey,
            observation: Observation,
            uniform_sample: bool = False,
            noise_scale: float = 0.0,
        ) -> Tuple[Action, Dict[str, ArrayTree]]:
            model_key, sample_key = jax.random.split(key)

            def uniform_action():
                return jax.random.uniform(
                    sample_key, self.action_space.shape, minval=self._action_space_low, maxval=self._action_space_high
                )

            def actor_action():
                actions = expand_apply(partial(self._actor.apply, actor_params, model_key))(observation)
                actions += jax.random.normal(key=sample_key, shape=actions.shape, dtype=actions.dtype) * noise_scale
                actions = jnp.clip(actions, self._action_space_low, self._action_space_high)
                return actions

            actions = jax.lax.cond(uniform_sample, uniform_action, actor_action)

            return actions, {}

        self._noise_schedule = optax.linear_schedule(
            self.config.initial_noise_scale, self.config.final_noise_scale, self.config.noise_scale_steps
        )

        self._actor_optimizer = optax.adam(self.config.actor_learning_rate)
        self._critic_optimizer = optax.adam(self.config.critic_learning_rate)

        self._rollout_fn = partial(rollout_truncated, env, 1, policy_fn)

        temp_key = jax.random.PRNGKey(0)
        temp_train_state = self._initial_train_state_without_replay_buffer(temp_key, temp_key)

        rollout_data_spec, _, _, _ = jax.eval_shape(
            self._rollout_fn,
            temp_train_state.params["online"]["actor"],
            temp_train_state.key,
            temp_train_state.time_step,
            temp_train_state.env_state,
            noise_scale=1.0,
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
        config = super(DDPG, DDPG).default_config()
        config.update(_DEFAULT_DDPG_CONFIG)
        return config

    def compute_action(
        self,
        params: hk.Params,
        key: PRNGKey,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        if not deterministic:
            raise NotImplementedError

        return self._actor.apply(params["online"], key, observation)

    def _initial_train_state_without_replay_buffer(
        self, consistent_key: PRNGKey, divergent_key: PRNGKey
    ) -> AnakinTrainState:
        train_state_key, env_key = jax.random.split(divergent_key)
        sample_obs = jnp.expand_dims(self.env.observation_space.sample(consistent_key), 0)
        sample_action = jnp.expand_dims(self.env.action_space.sample(consistent_key), 0)
        actor_params = self._actor.init(consistent_key, sample_obs)
        critic_params = self._critic.init(consistent_key, sample_obs, sample_action)
        params = {
            "online": {"actor": actor_params, "critic": critic_params},
            "target": {"actor": actor_params, "critic": critic_params},
        }
        time_step, env_state = self.env.reset(env_key)
        actor_opt_state = self._actor_optimizer.init(params["online"]["actor"])
        critic_opt_state = self._critic_optimizer.init(params["online"]["critic"])
        opt_state = {"actor": actor_opt_state, "critic": critic_opt_state}

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
        target_actor_params,
        key,
        obs,
        actions,
        rewards,
        next_obs,
        discounts,
        weights,
    ):
        target_actions = self._actor.apply(target_actor_params, key, next_obs)

        if self.config.smooth_target_policy:
            epsilon = jax.random.normal(key, target_actions.shape) * self.config.target_noise
            epsilon = jnp.clip(epsilon, -self.config.target_noise_clip, self.config.target_noise_clip)
            target_actions += epsilon
            target_actions = jnp.clip(
                target_actions,
                self._action_space_low,
                self._action_space_high,
            )

        target_pi_q_values = self._critic.apply(target_critic_params, key, next_obs, target_actions)

        if self.config.use_twin_critic:
            target_pi_q_values = jnp.minimum(*target_pi_q_values)

        backup = rewards + self.config.gamma * discounts * target_pi_q_values

        def loss_fn(a, b):
            if self.config.use_huber:
                return rlax.huber_loss(a - b)
            else:
                return rlax.l2_loss(a, b)

        if self.config.use_twin_critic:
            q1_values, q2_values = self._critic.apply(online_critic_params, key, obs, actions)
            q1_loss = loss_fn(backup, q1_values)
            q2_loss = loss_fn(backup, q2_values)
            critic_loss = 0.5 * (q1_loss + q2_loss)
            td_error = 0.5 * ((q1_values - backup) + (q2_values - backup))
        else:
            q_values = self._critic.apply(online_critic_params, key, obs, actions)
            critic_loss = loss_fn(backup, q_values)
            td_error = q_values - backup

        critic_loss = jnp.mean(critic_loss * weights)

        return critic_loss, {"critic_loss": critic_loss, "td_error": td_error}

    def _actor_loss(self, online_actor_params, online_critic_params, key, obs):
        pi_q_values = self._critic.apply(
            online_critic_params, key, obs, self._actor.apply(online_actor_params, key, obs)
        )

        if self.config.use_twin_critic:
            pi_q_values = jnp.minimum(*pi_q_values)

        actor_loss = -jnp.mean(pi_q_values)

        return actor_loss, {"actor_loss": actor_loss}

    def _update(self, train_step, params, opt_state, key, obs, actions, rewards, next_obs, discounts, weights):
        critic_grads, critic_info = jax.grad(self._critic_loss, has_aux=True)(
            params["online"]["critic"],
            params["target"]["critic"],
            params["target"]["actor"],
            key,
            obs,
            actions,
            rewards,
            next_obs,
            discounts,
            weights,
        )

        actor_grads, actor_info = jax.grad(self._actor_loss, has_aux=True)(
            params["online"]["actor"], params["online"]["critic"], key, obs
        )

        td_error = critic_info.pop("td_error")

        info = {"actor_loss": actor_info["actor_loss"], "critic_loss": critic_info["critic_loss"]}

        critic_grads, actor_grads, info = self._maybe_all_reduce("pmean", (critic_grads, actor_grads, info))

        critic_updates, new_critic_opt_state = self._critic_optimizer.update(
            critic_grads, opt_state["critic"], params["online"]["critic"]
        )
        actor_updates, new_actor_opt_state = self._actor_optimizer.update(
            actor_grads, opt_state["actor"], params["online"]["actor"]
        )

        new_online_critic_params = optax.apply_updates(params["online"]["critic"], critic_updates)
        new_online_actor_params = optax.apply_updates(params["online"]["actor"], actor_updates)

        new_online_actor_params, new_actor_opt_state = jax.tree.map(
            lambda a, b: jax.lax.select(
                jnp.logical_and(
                    jnp.any(train_step > 0),
                    jnp.any(train_step % self.config.policy_delay == 0),
                ),
                a,
                b,
            ),
            (new_online_actor_params, new_actor_opt_state),
            (params["online"]["actor"], opt_state["actor"]),
        )

        new_online_params = {"critic": new_online_critic_params, "actor": new_online_actor_params}

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
            new_online_params,
            params["target"],
        )

        new_params = {"online": new_online_params, "target": new_target_params}
        new_opt_state = {"actor": new_actor_opt_state, "critic": new_critic_opt_state}
        info = {}

        return new_params, new_opt_state, info, td_error

    def train_step(
        self, train_state: AnakinTrainStateWithReplayBuffer
    ) -> Tuple[AnakinTrainStateWithReplayBuffer, Metrics]:
        next_train_state_key, rollout_key, update_key, sample_key = jax.random.split(train_state.key, 4)

        noise_scale = self._noise_schedule(train_state.train_step)

        rollout_data, _, new_time_step, new_env_state = self._rollout_fn(
            train_state.params["online"]["actor"],
            rollout_key,
            train_state.time_step,
            train_state.env_state,
            uniform_sample=train_state.train_step <= self.config.random_steps,
            noise_scale=noise_scale,
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


@register("TD3")
class TD3(DDPG):
    @staticmethod
    def default_config() -> ConfigDict:
        config = super(TD3, TD3).default_config()
        config.update(
            {
                "use_twin_critic": True,
                "policy_delay": 2,
                "smooth_target_policy": True,
            }
        )
        return config

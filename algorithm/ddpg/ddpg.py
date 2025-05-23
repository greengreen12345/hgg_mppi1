from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium
from packaging import version

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from algorithm.replay_buffer import goal_based_process

from utils.tf_utils import Normalizer_torch

# fmt: off
# [start-config-dict-torch]
DDPG_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class DDPG(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Deep Deterministic Policy Gradient (DDPG)

        https://arxiv.org/abs/1509.02971

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(DDPG_DEFAULT_CONFIG)
        _cfg.update(vars(cfg) if cfg is not None else {})

        self.args = _cfg

        print("models：", models)

        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.target_policy = self.models.get("target_policy", None)
        self.critic = self.models.get("critic", None)
        self.target_critic = self.models.get("target_critic", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["target_policy"] = self.target_policy
        self.checkpoint_modules["critic"] = self.critic
        self.checkpoint_modules["target_critic"] = self.target_critic

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic is not None:
                self.critic.broadcast_parameters()

        if self.target_policy is not None and self.target_critic is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_policy.freeze_parameters(True)
            self.target_critic.freeze_parameters(True)

            # update target networks (hard update)
            self.target_policy.update_parameters(self.policy, polyak=1)
            self.target_critic.update_parameters(self.critic, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic is not None:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self._critic_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # Normalizer
        self.obs_normalizer = Normalizer_torch(16, self.device)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

        # clip noise bounds
        if self.action_space is not None:
            self.clip_actions_min = torch.tensor(self.action_space.low, device=self.device)
            self.clip_actions_max = torch.tensor(self.action_space.high, device=self.device)

        self.timestep = 0

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        if self.timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

        # add exloration noise
        if self._exploration_noise is not None:
            # sample noises
            noises = self._exploration_noise.sample(actions.shape)

            # define exploration timesteps
            scale = self._exploration_final_scale
            if self._exploration_timesteps is None:
                self._exploration_timesteps = timesteps

            # apply exploration noise
            if self.timestep <= self._exploration_timesteps:
                scale = (1 - self.timestep / self._exploration_timesteps) * (
                    self._exploration_initial_scale - self._exploration_final_scale
                ) + self._exploration_final_scale
                noises.mul_(scale)

                # modify actions
                actions.add_(noises)
                actions.clamp_(min=self.clip_actions_min, max=self.clip_actions_max)

                # record noises
                self.track_data("Exploration / Exploration noise (max)", torch.max(noises).item())
                self.track_data("Exploration / Exploration noise (min)", torch.min(noises).item())
                self.track_data("Exploration / Exploration noise (mean)", torch.mean(noises).item())

            else:
                # record noises
                self.track_data("Exploration / Exploration noise (max)", 0)
                self.track_data("Exploration / Exploration noise (min)", 0)
                self.track_data("Exploration / Exploration noise (mean)", 0)
        self.timestep = self.timestep + 1

        return actions, None, outputs


    def step(self, states: torch.Tensor, explore=False, goal_based=False):
        self.args['acts_dims'] = [self.action_space.shape[0]]

        if self.args['buffer'].steps_counter < self.args['warmup']:
            action = np.random.uniform(-0.08, 0.08, size=self.args['acts_dims'])

            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)

            return action

        if goal_based:
            states = goal_based_process(states)

        # eps-greedy exploration
        if explore and np.random.uniform() <= self.args['eps_act']:
            action = np.random.uniform(-0.08, 0.08, size=self.args['acts_dims'])

            if isinstance(action, np.ndarray):
                action = torch.tensor(action, dtype=torch.float32, device=self.device)

            return action

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        processed_states = self._state_preprocessor(states)

        if isinstance(processed_states, dict):
            processed_states = torch.cat(
                [torch.tensor(v, dtype=torch.float32, device=device).flatten() for v in processed_states.values()])
        states = torch.tensor(self._state_preprocessor(processed_states), dtype=torch.float32).to(device)

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            action, _, outputs = self.policy.act({"states": self._state_preprocessor(states)}, role="policy")

        # uncorrelated gaussian explorarion
        # if explore:
        #     noise = torch.tensor(
        #         np.random.normal(0, self.args['std_act'], size=self.args['acts_dims']),
        #         dtype=action.dtype,
        #         device=action.device
        #     )
        #     action = action + noise

            #action += np.random.normal(0, self.args['std_act'], size=self.args['acts_dims'])

        action = torch.clamp(action, -1.0, 1.0)

        print("RL action3")
        return action

    def preprocess_obs(self, obs):
        import numpy as np
        # Stitching all NumPy array values into a Tensor
        obs_tensor = torch.tensor(np.concatenate([
            obs['observation'].flatten(),
            obs['achieved_goal'].flatten(),
            obs['desired_goal'].flatten()
        ]), dtype=torch.float32)

        return obs_tensor

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                # compute target values
                with torch.no_grad():
                    next_actions, _, _ = self.target_policy.act({"states": sampled_next_states}, role="target_policy")

                    target_q_values, _, _ = self.target_critic.act(
                        {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic"
                    )
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_terminated | sampled_truncated).logical_not()
                        * target_q_values
                    )

                # compute critic loss
                critic_values, _, _ = self.critic.act(
                    {"states": sampled_states, "taken_actions": sampled_actions}, role="critic"
                )

                critic_loss = F.mse_loss(critic_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self._grad_norm_clip)

            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # compute policy (actor) loss
                actions, _, _ = self.policy.act({"states": sampled_states}, role="policy")
                critic_values, _, _ = self.critic.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic"
                )

                policy_loss = -critic_values.mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            self.track_data("Loss / Policy loss", policy_loss.item())
            self.track_data("Loss / Critic loss", critic_loss.item())

            self.track_data("Q-network / Q1 (max)", torch.max(critic_values).item())
            self.track_data("Q-network / Q1 (min)", torch.min(critic_values).item())
            self.track_data("Q-network / Q1 (mean)", torch.mean(critic_values).item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())

            if self._learning_rate_scheduler:
                self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])

    def target_update(self):
        """Perform a soft update of the target networks using Polyak averaging"""
        if self.target_policy is not None and self.target_critic is not None:
            self.target_policy.update_parameters(self.policy, polyak=self._polyak)
            self.target_critic.update_parameters(self.critic, polyak=self._polyak)

    def normalizer_update(self, batch):
        #print("obs.shape", np.array(batch['obs']).shape)
        #print("obs_next.shape", np.array(batch['obs_next']).shape)

        self.obs_normalizer.update(np.concatenate([batch['obs'], batch['obs_next']], axis=0))

    def train(self, batch):
        device = next(self.policy.parameters()).device
        #print(f"batch keys: {batch.keys()}")
        """Perform a training step using a batch of experience"""
        # Unpack batch data
        # states = batch["states"]
        # actions = batch["actions"]
        # rewards = batch["rewards"]
        # next_states = batch["next_states"]
        # terminated = batch["terminated"]
        # truncated = batch["truncated"]

        states = batch["obs"]
        actions = batch["acts"]
        # rewards = batch["rews"]
        next_states = batch["obs_next"]
        # terminated = batch["done"]
        # truncated = batch["done"]

        rewards = torch.tensor(batch["rews"], dtype=torch.float32).to(device).view(-1)
        terminated = torch.tensor(batch["done"], dtype=torch.bool).to(device).view(-1)

        if isinstance(next_states, list):
            states = torch.tensor(states, dtype=torch.float32)
            #actions = torch.tensor(actions, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
        elif isinstance(next_states, np.ndarray):
            states = torch.from_numpy(states).float()
            #actions = torch.from_numpy(actions).float()
            next_states = torch.from_numpy(next_states).float()

        states = states.to(device)
        #actions = actions.to(device)
        next_states = next_states.to(device)

        if isinstance(terminated, list):
            terminated = torch.tensor(terminated, dtype=torch.bool)
        elif isinstance(terminated, np.ndarray):
            terminated = torch.from_numpy(terminated).bool()
        terminated = terminated.to(device)

        # Process states
        states = self._state_preprocessor(states, train=True)
        next_states = self._state_preprocessor(next_states, train=True)

        # Compute target values
        with torch.no_grad():

            next_actions, _, _ = self.target_policy.act({"states": next_states.T}, role="target_policy")
            target_q_values, _, _ = self.target_critic.act(
                {"states": next_states.T, "taken_actions": next_actions}, role="target_critic"
            )
            target_q_values = target_q_values.view(-1)
            target_values = (
                    rewards
                    + self._discount_factor * terminated.logical_not() * target_q_values
            )

        # Compute critic loss
        if isinstance(states, list):
            states = torch.tensor(states, dtype=torch.float32)

        if isinstance(actions, list):
            if isinstance(actions[0], torch.Tensor):
                actions = torch.stack(actions)
            else:
                actions = torch.tensor(actions, dtype=torch.float32)

        critic_values, _, _ = self.critic.act(
            {"states": states.T, "taken_actions": actions}, role="critic"
        )
        critic_loss = F.mse_loss(critic_values, target_values)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.critic_optimizer)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self._grad_norm_clip)
        self.scaler.step(self.critic_optimizer)

        # Compute policy loss
        new_actions, _, _ = self.policy.act({"states": states.T}, role="policy")
        policy_q_values, _, _ = self.critic.act(
            {"states": states.T, "taken_actions": new_actions}, role="critic"
        )
        policy_loss = -policy_q_values.mean()

        # Optimize policy
        self.policy_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        if self._grad_norm_clip > 0:
            self.scaler.unscale_(self.policy_optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
        self.scaler.step(self.policy_optimizer)

        # Update the scaler (for mixed precision training)
        self.scaler.update()

        # Update target networks (soft update)
        self.target_policy.update_parameters(self.policy, polyak=self._polyak)
        self.target_critic.update_parameters(self.critic, polyak=self._polyak)

        # Collect training information
        info = {
            "policy_loss": policy_loss.item(),
            "critic_loss": critic_loss.item(),
            "q_value_mean": critic_values.mean().item(),
            "target_value_mean": target_values.mean().item(),
        }

        return info

    def get_q_value(self, obs: torch.Tensor) -> torch.Tensor:

        """Compute Q value for given observations using the current policy"""
        obs = self._state_preprocessor(obs, train=False)

        if isinstance(obs, list):
            obs = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        device = next(self.policy.parameters()).device
        obs = obs.to(device)

        with torch.no_grad():
            actions, _, _ = self.policy.act({"states": obs}, role="policy")
            q_values, _, _ = self.critic.act({"states": obs, "taken_actions": actions}, role="critic")
        return q_values[:, 0]





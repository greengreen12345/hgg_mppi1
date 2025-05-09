from skrl.envs.loaders.torch import load_isaacgym_env_preview4
from skrl.envs.wrappers.torch import wrap_env

import copy
import numpy as np
#from envs import make_env
from envs.utils import get_goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
import torch, hydra

from scripts.reactive_tamp import REACTIVE_TAMP
from src.m3p2i_aip.config.config_store import ExampleConfig
import  learner.utils.isaacgym_utils.isaacgym_wrapper as wrapper
from src.m3p2i_aip.utils.data_transfer import bytes_to_torch, torch_to_bytes
from sim1 import run_sim1

import random

class TrajectoryPool:
	def __init__(self, args, pool_length):
		self.args = args
		self.length = pool_length

		self.pool = []
		self.pool_init_state = []
		self.counter = 0

	def insert(self, trajectory, init_state):
		if self.counter<self.length:
			self.pool.append(trajectory.copy())
			self.pool_init_state.append(init_state.copy())
		else:
			self.pool[self.counter%self.length] = trajectory.copy()
			self.pool_init_state[self.counter%self.length] = init_state.copy()
		self.counter += 1

	def pad(self):
		if self.counter>=self.length:
			return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
		pool = copy.deepcopy(self.pool)
		pool_init_state = copy.deepcopy(self.pool_init_state)
		while len(pool)<self.length:
			pool += copy.deepcopy(self.pool)
			pool_init_state += copy.deepcopy(self.pool_init_state)
		return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])

	def clear(self):
		"""Clear all trajectories and states"""
		self.pool.clear()
		self.pool_init_state.clear()
		self.counter = 0

class MatchSampler:
    def __init__(self, args, traj_pool, env):
        self.args = args
        self.traj_pool = traj_pool
        self.goal_distance = lambda a, b: np.linalg.norm(a - b)
        self.env = env

        # Initialize history of best goal and best score
        self.last_best_goal = None
        self.last_best_score = float("inf")

        # Initialize current desired goal from the environment (only once at init)
        current_obs = self.env.reset()
        self.current_desired = current_obs['desired_goal'].copy()

    def reset(self):
        """Reset history of best goal and best distance"""
        self.last_best_goal = None
        self.last_best_score = float("inf")

    def sample(self):
        print(f"Before learning: {self.traj_pool.counter} trajectories in pool.")

        # If the trajectory pool is empty, return the current desired goal
        if len(self.traj_pool.pool) == 0:
            return self.current_desired.copy()

        # Randomly sample up to 32 trajectories from the pool
        sampled_trajs = random.sample(self.traj_pool.pool, min(32, len(self.traj_pool.pool)))

        best_candidate = None
        best_candidate_score = float("inf")

		#Search for the achieved_goal that is closest to current desired_goal
        for traj in sampled_trajs:
            for goal in traj:
                dist = self.goal_distance(goal, self.current_desired)
                if dist < best_candidate_score:
                    best_candidate = goal
                    best_candidate_score = dist

        # Compare with historical best and update if improved
        if best_candidate_score < self.last_best_score:
            self.last_best_goal = best_candidate.copy()
            self.last_best_score = best_candidate_score
            print(f"[MatchSampler] New best goal updated with dist = {best_candidate_score:.3f}")
        else:
            if hasattr(self.args, 'verbose') and self.args.verbose:
                print(f"[MatchSampler] Keeping previous best goal with dist = {self.last_best_score:.3f}")

        return self.last_best_goal.copy()

class HGGLearner:
	def __init__(self, args):
		self.args = args
		self.goal_distance = get_goal_distance(args)

		#self.env_List = []
		self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)

		self.sampler = None
		self.reactive_tamp = None

	def learn(self, args, env, env_test, agent, buffer, planner):
		
		self.achieved_trajectory_pool.clear()

		self.initial_goals = []
		self.desired_goals = []
		self.explore_goals = []
		self.achieved_trajectories = []
		self.achieved_rewards = []

		self.env = env
		self.env_test = env_test

		if self.sampler is None:
			self.sampler = MatchSampler(args, self.achieved_trajectory_pool, self.env)
		self.sampler.reset()

		initial_goals = []
		desired_goals = []

		obs = self.env.reset()
		goal_a = obs['achieved_goal'].copy()
		goal_d = obs['desired_goal'].copy()
		initial_goals.append(goal_a.copy())
		desired_goals.append(goal_d.copy())

		self.initial_goals = initial_goals
		self.desired_goals = desired_goals

		achieved_trajectories = []
		achieved_init_states = []
		achieved_rewards = []

		init_state = obs['observation'].copy()
		self.episode_return = 0

		for i in range(args.episodes):

			sampled_goal = self.sampler.sample()
			action_hgg = agent.step(obs, explore=True, goal_based=True)
			print("**************sampled_goal***********", sampled_goal)
			print("***********action***********", action_hgg)

			sampled_goal_tensor = torch.tensor(sampled_goal, device=action_hgg.device, dtype=action_hgg.dtype)
			explore_goal = sampled_goal_tensor + action_hgg
			self.explore_goals.append(explore_goal)
			self.env.sampled_goal = sampled_goal_tensor
			self.env.goal = explore_goal

			obs = self.env._get_obs()
			current = Trajectory(obs)
			trajectory = [obs['achieved_goal'].copy()]

			total_reward = 0

			if i == 0:
				args.timesteps = 10

			for timestep in range(args.timesteps):

				action_mppi = bytes_to_torch(
					planner.run_tamp(
						torch_to_bytes(self.env._dof_state), torch_to_bytes(self.env._root_state), explore_goal.tolist())
				)

				obs, reward, done, info, dis = self.env.step(action_mppi)
				total_reward += reward

				trajectory.append(obs['achieved_goal'].copy())

				if dis < 0.01:
					print("**********done*************")
					break

			action = action_hgg
			self.episode_return += total_reward
			current.store_step(action, obs, total_reward, done)

			# insert every trajectory into the achieved_trajectory_pool
			self.achieved_trajectory_pool.insert(np.array(trajectory).copy(), init_state.copy())

			# store every trajectory into the achieved_trajectories for plotting
			achieved_trajectories.append(np.array(trajectory))
			self.achieved_trajectories = achieved_trajectories
			achieved_init_states.append(init_state)

			# store every step's current(action, obs, total_reward, done) into buffer
			buffer.store_trajectory(current)

			agent.normalizer_update(buffer.sample_batch())

			if buffer.steps_counter>=args.warmup:
				for _ in range(args.train_batches):
					info = agent.train(buffer.sample_batch())
					#args.logger.add_dict(info)
				agent.target_update()






from pathlib import Path

import gym
from gym.spaces import Tuple
import torch
from .ddpg import DDPG

import numpy as np


class FrozenTag(gym.Wrapper):
    """Tag with frozen prey agent"""

    def __init__(self, env, num_predators, num_preys):
        super().__init__(env)

        self.num_predators = num_predators
        self.num_preys = num_preys

        self.prey_action_space = self.action_space[-1]
        self.prey_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:num_predators])
        self.observation_space = Tuple(self.observation_space[:num_predators])
        
        self.n_agents = num_predators
        self.unwrapped.n_agents = num_predators

    def reset(self):
        obs = super().reset()
        return obs[:self.num_predators]

    def step(self, action):
        frozen_action = [0 for _ in range(self.num_preys)]
        action = tuple(action) + tuple(frozen_action)
        obs, rew, done, info = super().step(action)
        obs = obs[:self.num_predators]
        rew = rew[:self.num_predators]
        return obs, rew, done, info


class RandomTag(gym.Wrapper):
    """Tag with random prey agent"""

    def __init__(self, env, num_predators, num_preys):
        super().__init__(env)

        self.num_predators = num_predators
        self.num_preys = num_preys

        self.prey_action_space = self.action_space[-1]
        self.prey_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:num_predators])
        self.observation_space = Tuple(self.observation_space[:num_predators])

        self.n_agents = num_predators
        self.unwrapped.n_agents = num_predators

    def reset(self):
        obs = super().reset()
        return obs[:self.num_predators]

    def step(self, action):
        random_action = [self.prey_action_space.sample() for _ in range(self.num_preys)]
        action = tuple(action) + tuple(random_action)
        obs, rew, done, info = super().step(action)
        obs = obs[:self.num_predators]
        rew = rew[:self.num_predators]
        return obs, rew, done, info


class PretrainedTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, env, num_predators, num_preys):
        super().__init__(env)

        self.max_predators_4_preys = 3

        self.num_predators = num_predators
        self.num_preys = num_preys

        self.predators = self.env.unwrapped.agents[:self.num_predators]
        self.preys = self.env.unwrapped.agents[self.num_predators:]
        self.landmarks = self.env.unwrapped.world.landmarks
        
        self.prey_action_space = self.action_space[-1]
        self.prey_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:num_predators])
        self.observation_space = Tuple(self.observation_space[:num_predators])

        self.n_agents = num_predators
        self.unwrapped.n_agents = num_predators

        self.prey_policy = DDPG(14, 5, 50, 128, 0.01)
        # current file dir
        param_path = Path(__file__).parent / "prey_params.pt"
        save_dict = torch.load(param_path)
        self.prey_policy.load_params(save_dict["agent_params"][-1])
        self.prey_policy.policy.eval()

    def reset(self):
        obs = super().reset()
        return obs[:self.num_predators]

    def step(self, action):

        prey_obs = self._get_prey_obs()
        prey_actions = self.prey_policy.step(prey_obs)

        action = tuple(action) + tuple(prey_actions,)
        obs, rew, done, info = super().step(action)

        obs = obs[:self.num_predators]
        rew = rew[:self.num_predators]
        return obs, rew, done, info

    def _get_prey_obs(self):
        prey_pos = np.array([prey.state.p_pos for prey in self.preys])
        prey_vel = np.array([prey.state.p_vel for prey in self.preys])
        
        landmark_pos = np.array([landmark.state.p_pos for landmark in self.landmarks if not landmark.boundary])
        landmark_relative = landmark_pos[None, :, :] - prey_pos[:, None, :]  # (n_preys, n_landmarks, 2)
        
        predator_pos = np.array([predator.state.p_pos for predator in self.predators])
        dists = np.linalg.norm(predator_pos[None, :, :] - prey_pos[:, None, :], axis=2)  # (n_preys, n_predators)
        
        sorted_indices = np.argsort(dists, axis=1)[:, :self.max_predators_4_preys]
        nearest_predator_pos = predator_pos[sorted_indices] - prey_pos[:, None, :]  # (n_preys, max_pred, 2)
        
        obs = np.concatenate([
            prey_vel,  # (n_preys, 2)
            prey_pos,  # (n_preys, 2)
            landmark_relative.reshape(len(self.preys), -1),  # (n_preys, n_landmarks*2)
            nearest_predator_pos.reshape(len(self.preys), -1)  # (n_preys, max_pred*2)
        ], axis=1)
        
        return obs
    
class RuleTag(gym.Wrapper):
    """Tag with rule based prey agent"""

    def __init__(self, env, num_predators, num_preys):
        super().__init__(env)

        self.num_predators = num_predators
        self.num_preys = num_preys
        self.env = env

        self.predators = self.env.unwrapped.agents[:self.num_predators]
        self.preys = self.env.unwrapped.agents[self.num_predators:]

        self.prey_action_space = self.action_space[-1]
        self.prey_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:num_predators])
        self.observation_space = Tuple(self.observation_space[:num_predators])
        
        self.n_agents = num_predators
        self.unwrapped.n_agents = num_predators

    def reset(self):
        obs = super().reset()
        return obs[:self.num_predators]

    def step(self, action):
        # The action of each prey is the action of its nearest predator.
        rule_action = []
        predator_pos = np.array([predator.state.p_pos for predator in self.predators])
        for prey in self.preys:
            prey_pos = prey.state.p_pos
            nearest_predator_idx = np.argmin(np.linalg.norm(prey_pos - predator_pos, axis=1))
            rule_action.append(action[nearest_predator_idx])
            
        action = tuple(action) + tuple(rule_action)
        obs, rew, done, info = super().step(action)
        obs = obs[:self.num_predators]
        rew = rew[:self.num_predators]
        return obs, rew, done, info

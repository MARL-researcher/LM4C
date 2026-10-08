import importlib
from .environment import MultiAgentEnv
from gym.spaces import flatdim
import numpy as np
from envs.common_wrappers import TimeLimit, FlattenObservation
import envs.mpe.pretrained as pretrained
from absl import logging

class MPEWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):

        self.episode_limit = time_limit

        scenario = importlib.import_module("envs.mpe.scenarios."+key).Scenario()
        world = scenario.make_world(kwargs)

        self.reward_individual = False
        if kwargs.get("individual_reward", False):
            self.reward_individual = True

        # get visibility range
        if kwargs.get("visibility_range", False):
            self.visibility_range = kwargs["visibility_range"]

        # set the reward
        if kwargs.get("reward_discrete", False) and hasattr(scenario, "reward_discrete"):
            reward = scenario.reward_discrete
        elif kwargs.get("reward_rendundant", False) and hasattr(scenario, "reward_rendundant"):
            reward = scenario.reward_rendundant
        elif kwargs.get("individual_reward", False) and hasattr(scenario, "reward_individual"):
            reward = scenario.reward_individual
        else:
            reward = scenario.reward

        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=reward,
            observation_callback=(
                scenario.entity_observation if kwargs.get("obs_entity_mode", False) else 
                scenario.observation
            ),
            state_callback=(
                scenario.entity_state if kwargs.get("state_entity_mode", False) else
                None
            ),
            world_info_callback=getattr(scenario, "world_benchmark_data", None)
        )

        # TODO: a warning is introduced here but the env is still working
        self._env = TimeLimit(env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if kwargs.get("prey_wrapper", False):
            prey_options = dict(num_predators=kwargs["num_predators"], num_preys=kwargs["num_preys"])
            if kwargs["prey_wrapper"] == "PretrainedTag":
                prey_options["device"] = kwargs.get("prey_device")
            self._env = getattr(pretrained, kwargs["prey_wrapper"])(self._env, **prey_options)

        # basic env variables
        self.n_agents    = self._env.unwrapped.n_agents
        self.n_landmarks = len(world.landmarks)
        self.n_entities  = getattr(scenario, "num_entities", self.n_agents+self.n_landmarks)
        self.n_entity_types = getattr(scenario, "num_entity_types", 2)
        self._obs = None
        self._state = None

        self.obs_entity_feats = getattr(scenario, "entity_obs_feats", 0)
        self.state_entity_feats = getattr(scenario, "entity_state_feats", 0)

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        # check if the scenario uses an entity state 
        self.custom_state = kwargs.get("state_entity_mode", False)
        self.custom_state_dim = self.state_entity_feats*self.n_entities

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

        # variables for animation
        self.t = 0
        # Keep all world agents (including pretrained prey), the initial state
        # and the terminal state. Policy-agent counts remain unchanged.
        self.agent_positions = np.zeros((len(world.agents), self.episode_limit + 1, 2))
        self.landmark_positions = np.zeros((len(world.landmarks), self.episode_limit + 1, 2))
        self._animation_frames = 0
        self.agent_colors = [a.color for a in world.agents]
        self.landmark_colors      = [l.color for l in world.landmarks]
        self.agent_sizes = [a.marker_size for a in world.agents]
        self.landmark_sizes       = [l.marker_size for l in world.landmarks]
        self.episode_rewards_all = np.zeros(self.episode_limit + 1)
        if self.reward_individual:
            self.episode_rewards_indiv = np.zeros((self.episode_limit + 1, self.n_agents))

    def step(self, actions, animate=False):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        # the reward is devided by the eposode length and the numbers of agents
        for i in range(len(reward)):
            reward[i] /= (self.episode_limit * self.n_agents)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.custom_state:
            self._state = self._env.get_state()

        # save step description if animation is required
        if animate:
            self._record_animation_frame(self.t + 1)
            if self.reward_individual:
                self.episode_rewards_indiv[self.t + 1,:] = reward
            self.episode_rewards_all[self.t + 1] = float(sum(reward))
        self.t += 1

        if self.reward_individual:
            return reward, all(done), info["world"]
        else:
            return float(sum(reward)), all(done), info["world"]

    def get_agent_positions(self):
        """Returns the [x,y] positions of each agent in a list"""
        return [a.state.p_pos for a in self._env.world.agents]

    def get_landmark_positions(self):
        return [l.state.p_pos for l in self._env.world.landmarks]

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        if self.custom_state:
            return self._state
        else: 
            return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.custom_state:
            return self.custom_state_dim
        else:
            return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)
    
    def get_visibility_matrix(self):
        """Returns visibility matrix, common for all scenarios"""
        # agent_list = self._env.unwrapped.agents
        # pos_arr = np.array([agent_list[i].state.p_pos for i in range(self.n_agents)])
        # diff = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
        # dist_matrix = np.linalg.norm(diff, axis=-1)
        # neighbors_matrix = (dist_matrix < self.visibility_range).astype(np.int32)

        """new logic for algorithm LAIES in scenario PP"""
        agent_list = self._env.unwrapped.agents
        num_predators = self._env.num_predators
        num_preys = self._env.num_preys

        all_positions = np.array([agent.state.p_pos for agent in agent_list])
        predator_pos = all_positions[:num_predators]
        prey_pos = all_positions[num_predators:]

        diff_x = predator_pos[:, 0, np.newaxis] - prey_pos[np.newaxis, :, 0]
        diff_y = predator_pos[:, 1, np.newaxis] - prey_pos[np.newaxis, :, 1]
        dist_sq = diff_x**2 + diff_y**2

        closest_indices = np.argpartition(dist_sq, 2, axis=1)[:, :2]

        prey_visibility = np.zeros((num_predators, num_preys), dtype=np.int32)
        rows = np.repeat(np.arange(num_predators), 2)
        cols = closest_indices.flatten()
        prey_visibility[rows, cols] = 1

        visibility_matrix = np.hstack([np.ones((num_predators, num_predators), dtype=np.int32), prey_visibility])

        return visibility_matrix
    
    def get_extrinsic_state(self):
        """Returns extrinsic state for LAIES"""
        agent_list = self._env.unwrapped.agents[self._env.num_predators:]
        enemy_state = np.zeros((self._env.num_preys, 4))
        for i in range(self._env.num_preys):
            enemy_state[i][0] = agent_list[i].state.p_pos[0]
            enemy_state[i][1] = agent_list[i].state.p_pos[1]
            enemy_state[i][2] = agent_list[i].state.p_vel[0]
            enemy_state[i][3] = agent_list[i].state.p_vel[1]
        enemy_state = enemy_state.flatten().astype(np.float32)
        return enemy_state
    
    def get_alive_agent(self):
        return np.array([1. for _ in range(self.n_agents)])

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        if self.custom_state:
            self._state = self._env.get_state()
        self.t = 0
        self.agent_positions.fill(0.)
        self.landmark_positions.fill(0.)
        self.episode_rewards_all.fill(0.)
        if self.reward_individual:
            self.episode_rewards_indiv.fill(0.)
        self._animation_frames = 0
        self._record_animation_frame(0)
        return self.get_obs(), self.get_state()

    def _record_animation_frame(self, frame):
        self.agent_positions[:, frame, :] = np.asarray(self.get_agent_positions()).reshape(-1, 2)
        self.landmark_positions[:, frame, :] = np.asarray(self.get_landmark_positions()).reshape(-1, 2)
        self._animation_frames = frame + 1

    def render(self):
        return self._env.render(mode='rgb_array')

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def snapshot_animation(self):
        """Keep a trajectory independent of the buffers cleared by reset()."""
        frames = self._animation_frames
        return dict(
            agent_positions=self.agent_positions[:, :frames].copy(),
            landmark_positions=self.landmark_positions[:, :frames].copy(),
            episode_rewards=self.episode_rewards_all[:frames].copy(),
        )

    def create_animator(self, episode=None, start_frame=0, **options):
        """Build a CN/PP renderer from the recorded episode and actual world."""
        from matplotlib import pyplot as plt
        from .animate.icons import make_icon
        from .animate.pyplot_animator import MPEAnimator as CNAnimator
        from .animate.pyplot_animator_for_pp import MPEAnimator as PPAnimator

        world = self._env.world
        is_pp = any(getattr(agent, "adversary", False) for agent in world.agents)
        agent_colors, agent_icons, agent_labels = [], [], []
        predator_id = prey_id = 0
        palette = plt.get_cmap("tab20")
        for index, agent in enumerate(world.agents):
            if is_pp:
                if getattr(agent, "adversary", False):
                    color, kind = (0.90, 0.43, 0.13), "tiger"
                    label = f"Predator {predator_id + 1}"
                    predator_id += 1
                else:
                    color, kind = (0.25, 0.62, 0.42), "deer"
                    label = f"Prey {prey_id + 1}"
                    prey_id += 1
                icon = make_icon(kind)
            else:
                color = palette(index % 20)[:3]
                label = f"Robot {index + 1}"
                icon = make_icon("robot", color)
            agent_colors.append(color)
            agent_icons.append(icon)
            agent_labels.append(label)

        metadata = dict(
            agent_icons=agent_icons,
            landmark_icons=[make_icon("rock" if is_pp else "target") for _ in world.landmarks],
            agent_world_sizes=[agent.size for agent in world.agents],
            landmark_world_sizes=[landmark.size for landmark in world.landmarks],
            agent_labels=agent_labels,
            landmark_labels=[f"{'Obstacle' if is_pp else 'Goal'} {index + 1}" for index in range(len(world.landmarks))],
            scene_title="Predator–Prey" if is_pp else "Cooperative Navigation",
            frame_offset=start_frame,
        )
        metadata.update(options)
        animator = PPAnimator if is_pp else CNAnimator
        if episode is None:
            episode = dict(
                agent_positions=self.agent_positions[:, :self._animation_frames],
                landmark_positions=self.landmark_positions[:, :self._animation_frames],
                episode_rewards=self.episode_rewards_all[:self._animation_frames],
            )
        if not 0 <= start_frame < episode["agent_positions"].shape[1]:
            raise ValueError("start_frame must identify a recorded frame")
        return animator(
            episode["agent_positions"][:, start_frame:],
            episode["landmark_positions"][:, start_frame:],
            episode["episode_rewards"][start_frame:],
            agent_colors, self.landmark_colors, self.agent_sizes, self.landmark_sizes,
            **metadata,
        )

    def save_animation(self, path, **options):
        from matplotlib import pyplot as plt
        anim = self.create_animator(**options)
        try:
            return anim.save_animation(path)
        finally:
            plt.close(anim.fig)

    def get_stats(self):
        return {}
    
    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        for x in [
            "obs_entity_feats",
            "state_entity_feats",
            "n_entities",
            "n_entities_obs",
            "n_entities_state"
        ]:
            self.check_add_attribute(env_info, x)
        return env_info
    
    def get_ext_info(self):
        # new for algorithm LAIES
        extra_inf_size = self._env.num_preys * 4
        n_enemies = self._env.num_preys

        return extra_inf_size, n_enemies
    
    def check_add_attribute(self, info, attribute):
        if hasattr(self, attribute):
            info[attribute] = getattr(self, attribute)
        elif attribute in {"n_entities_obs", "n_entities_state"}: 
            pass 
        else:
            logging.warning(f"To use transformers you should define the {attribute} attribute in your environment __init__")

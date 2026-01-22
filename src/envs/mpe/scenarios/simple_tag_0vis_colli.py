import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        # set any world properties first
        self.shape = True if args is None else args["shape"]
        world.dim_c = 2
        num_good_agents = 1 if args is None else args["num_preys"]
        num_adversaries = 3 if args is None else args["num_predators"]
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 2 if args is None else args["num_landmarks"]
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        predator_index = 0
        prey_index = 0
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            if agent.adversary:
                agent.name = 'predator%d' % predator_index
                agent.size = 0.075 if args is None else args["predator_size"]
                agent.accel = 3.0 if args is None else args["predator_accel"]
                predator_index += 1
            else:
                agent.name = 'prey%d' % prey_index
                agent.size = 0.05 if args is None else args["prey_size"]
                agent.accel = 4.0 if args is None else args["prey_accel"]
                prey_index += 1
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):

        constant_size = 12
        for agent in world.agents:
            if agent.adversary:
                anchor_size = agent.size
                break

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            agent.marker_size = constant_size * (agent.size / anchor_size) if not agent.adversary else constant_size
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.marker_size = constant_size * (landmark.size / anchor_size)
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-2, +2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = world.np_random.uniform(-2, +2, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):

        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = self.shape
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world, ratio=[0.25, 0.5, 0.25]):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = self.shape
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * ratio[0] * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10 * ratio[1]
            # decreased reward for collision of adversaries
            for _adv in adversaries:
                if agent is _adv: continue
                if self.is_collision(agent, _adv):
                    rew -= 1 * ratio[2]
        return rew

    def observation(self, agent, world):

        agent_pos = agent.state.p_pos
        agent_vel = agent.state.p_vel

        landmarks_pos = np.array([e.state.p_pos for e in world.landmarks if not e.boundary])
        entity_pos = (landmarks_pos - agent_pos).flatten()
        
        other_agents = [a for a in world.agents if a is not agent]
        other_pos_all = np.array([a.state.p_pos for a in other_agents]) - agent_pos
        other_vel_all = np.array([a.state.p_vel for a in other_agents])
        other_adv = np.array([a.adversary for a in other_agents])
        
        if agent.adversary:
            non_adv_mask = ~other_adv
            dist_sq = np.sum(other_pos_all[non_adv_mask]**2, axis=1)
            closest_idx = np.argpartition(dist_sq, 2)[:2]
            
            other_pos = other_pos_all[non_adv_mask][closest_idx]
            other_vel = other_vel_all[non_adv_mask][closest_idx]
        else:
            other_pos = other_pos_all
            other_vel = other_vel_all[~other_adv]
        
        return np.concatenate([
            agent_vel,
            agent_pos,
            entity_pos,
            other_pos.flatten(),
            other_vel.flatten()
        ])

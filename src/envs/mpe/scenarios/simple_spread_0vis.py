import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        self.num_agents = 3 if args is None else args["num_agents"]
        self.num_landmarks = 3 if args is None else args["num_landmarks"]
        self.num_entities = self.num_agents+self.num_landmarks
        self.num_entity_types = 3 # agent, landmark, self
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

            agent.size = 0.15 if args is None else args["agent_size"]
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        self.world_benchmark = {'collisions':0, 'min_dists':0}
        
        # to acess a priori the dimension of the entity state
        self.entity_obs_feats = 4
        self.entity_state_feats = 5
        self.benchmark_circles = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.individual_reward = False if (args is None) else (args["individual_reward"])

        return world

    def reset_world(self, world):

        constant_size = 12
        anchor_size = world.agents[0].size

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])
            agent.marker_size = constant_size
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.marker_size = constant_size * (landmark.size / anchor_size)
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-2.5, +2.5, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = world.np_random.uniform(-2.5, +2.5, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    
    def world_benchmark_data(self, world, final=False):
        info = self.world_benchmark.copy()
        if final:
            for c in self.benchmark_circles:
                info[f'occupied_landmarks_{c}'] = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            info['min_dists'] += min(dists)
            if final:
                for c in self.benchmark_circles:
                    if min(dists) <= c:
                        info[f'occupied_landmarks_{c}'] += 1
        if final:
            for c in self.benchmark_circles:
                info[f'occupied_landmarks_{c}'] /= len(world.landmarks)

        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if a is not agent and self.is_collision(a, agent):
                        info["collisions"] += 1
        return info

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward_discrete(self, agent, world):
        """
        Discretized global reward using a distance threshold to reward agents close enough to landmarks
        """
        if agent.name != "agent 0":
            return 0

        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.1:
                rew += 1
        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if a is not agent and self.is_collision(a, agent):
                        rew -= 0.1
        return rew


    def reward(self, agent, world):
        """
        A global reward equivalent to PettingZoo, modelled by rewarding only one agent
        """
        if agent.name != "agent 0":
            return 0

        rew = 0
        # The lowest penalty is only incurred when all landmarks are occupied.
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
            
        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if a is not agent and self.is_collision(a, agent):
                        rew -= 1
        return rew

    def reward_rendundant(self, agent, world):
        """
        This is the original reward, which is the same for all the agents and is therefore rendundant
        when summed up. 
        """
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # The lowest penalty is only incurred when all landmarks are occupied.
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew# this second level of rendundancy was originally present in environment.py
    

    def reward_individual(self, agent, world, ratio=0.5):
        """
        Each agent gets its own reward based on the distance to the closest landmark
        """
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists) * (1-ratio)
        if agent.collide:
            for a in world.agents:
                if agent is a: continue
                if self.is_collision(a, agent):
                    rew -= 1 * ratio
        return rew

    def entity_state(self, world):

        """
        Entity approach for transformers including velocity

        *5 features for each entity: (pos_x, pos_y, vel_x, vel_y, is_agent)*
        - agent features: (x, y, vel_x, vel_y, 1)
        - landmark features: (x, y, vel_x, vel_y, 0)
        
        In this case the absolute position of the agents is considered,
        and the velocity is also included. All these informations are present
        alreay in the original state vector, because are included in the agents 
        observations which are concatenated together. 
        """

        feats = np.zeros(self.entity_state_feats*self.num_entities)
        
        # agents features
        i = 0
        for a in world.agents:
            pos = a.state.p_pos
            vel = a.state.p_vel
            feats[i:i+self.entity_state_feats] = [pos[0],pos[1],vel[0],vel[1],1.]
            i += self.entity_state_feats

        # landmarks features
        for landmark in world.landmarks:
            pos = landmark.state.p_pos
            vel = landmark.state.p_vel # the velocity in this case is just 0
            feats[i:i+self.entity_state_feats] = [pos[0],pos[1],vel[0],vel[1],0.]
            i += self.entity_state_feats

        return feats

    def entity_observation(self, agent, world):

        """
        Entity approach for transformers

        *4 features for each entity: (rel_x, rel_y, is_agent, is_self)*
        - rel_x and rel_y are the relative positions of an entity in respect to agent
        - communication is not considered because is not present in this scenario
        - velocity is not included (since in the original observaion an agent doesn't know other agents' velocities)

        Ex: agent 1 for agent 1:    (0, 0, 1, 1)
        Ex: agent 2 for agent 1:    (dx(a1,a2), dy(a1,a2), 1, 0)
        Ex: landmark 1 for agent 1: (dx(a1,l1), dy(a1,l1), 0, 0)
        
        """
    
        feats = np.zeros(self.entity_obs_feats*self.num_entities)
        
        # agent features
        pos_a = agent.state.p_pos
        i = 0
        for a in world.agents:
            if a is agent:
                #vel = agent.state.p_vel
                feats[i:i+self.entity_obs_feats] = [0., 0., 1., 1.]
            else:
                pos = a.state.p_pos - pos_a
                feats[i:i+self.entity_obs_feats] = [pos[0], pos[1], 1.,0.]
            i += self.entity_obs_feats

        # landmarks features
        for j, landmark in enumerate(world.landmarks):
            pos = landmark.state.p_pos - pos_a
            feats[i:i+self.entity_obs_feats] = [pos[0],pos[1], 0., 0.]
            i += self.entity_obs_feats

        feats = np.zeros(self.entity_obs_feats * self.num_entities)
        
        pos_a = agent.state.p_pos
        
        agents_pos = np.array([a.state.p_pos for a in world.agents])
        rel_agents_pos = agents_pos - pos_a
        
        agents_feats = np.zeros((len(world.agents), self.entity_obs_feats))
        agents_feats[:, :2] = rel_agents_pos
        agents_feats[:, 2] = 1  # is_agent=1
        
        self_idx = world.agents.index(agent)
        agents_feats[self_idx, 3] = 1  # is_self=1
        
        landmarks_pos = np.array([l.state.p_pos for l in world.landmarks])
        rel_landmarks_pos = landmarks_pos - pos_a
        
        landmarks_feats = np.zeros((len(world.landmarks), self.entity_obs_feats))
        landmarks_feats[:, :2] = rel_landmarks_pos
        
        all_feats = np.vstack([agents_feats, landmarks_feats])
        
        assert len(all_feats) == self.num_entities, \
            f"dismatch: expected: {self.num_entities}, factual: {len(all_feats)}"

        return feats

    
    def observation(self, agent, world):

        landmarks_pos = np.array([entity.state.p_pos for entity in world.landmarks])
        
        relative_pos = landmarks_pos - agent.state.p_pos
        
        dist_sq = np.sum(relative_pos**2, axis=1)
        
        closest_idx = np.argpartition(dist_sq, 2)[:2]
        
        return np.concatenate([
            agent.state.p_vel,
            agent.state.p_pos,
            relative_pos[closest_idx[0]],
            relative_pos[closest_idx[1]]
        ])

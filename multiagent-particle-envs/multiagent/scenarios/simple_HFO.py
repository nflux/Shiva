import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import hfo 

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        
        world.hfo_env = hfo.HFOEnvironment()
        world.hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
                            config_dir='/home/mehrzad/Desktop/work/PlayaVista/Sigma-Robocup/bin/teams/base/config/formations-dt', 
                        server_port=6000, server_addr='localhost', team_name='base_left', play_goalie=False)
         
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(1)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
         
        return world

    
    def reset_world(self, world):
        
        # random properties for agents
        '''for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        '''
    def reward(self, agent, world):
        
        return -1.0 
    
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        print('querying agents observation : ', world.hfo_env.getState() )
        return world.hfo_env.getState()
        
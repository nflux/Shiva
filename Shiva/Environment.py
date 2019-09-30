import gym

def initialize_env(env_params):

    if env_params['env_type'] == 'Gym':
        env = GymEnvironment(env_params['environment'],env_params['num_agents'],env_params['env_render'])

    return env


class Environment():
    def __init__(self, environment,num_agents):
        self.env = environment
        self.num_agents = num_agents
        self.obs = [0 for i in range(num_agents)]
        self.acs = [0 for i in range(num_agents)]
        self.rews = [0 for i in range(num_agents)]
        self.world_status = [0 for i in range(num_agents)]
        self.observation_space = None
        self.action_space = None
        self.step_count = 0

    def step(self,actions):
        pass

    def get_observation(self,agent_idx):
        return self.obs[agent_idx]

    def get_observations(self):
        return self.obs

    def get_action(self,agent_idx):
        return self.acs[agent_idx]

    def get_actions(self):
        return self.acs

    def get_reward(self,agent_idx):
        return self.rews[agent_idx]

    def get_rewards(self):
        return self.rews

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_current_step(self):
        return self.step_count

    def reset(self):
        pass

    def load_viewer(self):
        pass



class GymEnvironment(Environment):
    def __init__(self,environment,num_agents,render):

        #super(GymEnvironment,self).__init__(environment,num_agents)
        self.env = gym.make(environment)
        self.num_agents = num_agents
        self.obs = [0 for i in range(num_agents)]
        self.acs = [0 for i in range(num_agents)]
        self.rews = [0 for i in range(num_agents)]
        self.world_status = [0 for i in range(num_agents)]
        self.observation_space = self.env.observation_space.shape if self.env.observation_space.shape != () else self.env.observation_space.n
        self.action_space = self.env.action_space.shape if self.env.action_space.shape != () else self.env.action_space.n
        self.step_count = 0
        if render:
            self.load_viewer()


    def step(self,actions):

        for i in range(self.num_agents):
            self.acs = actions
            self.obs[i],self.rews[i],self.world_status[i], info = self.env.step(actions[i])

            if self.world_status[i]:
                self.world_status[i] = 'Episode Complete'
            else:
                self.world_status[i] = 'Episode In Progress'

        self.step_count +=1

        return self.obs,self.rews,self.world_status 

    def reset(self):
        for i in range(self.num_agents):
            self.obs[i] = self.env.reset()

    def load_viewer(self):
        self.env.render()

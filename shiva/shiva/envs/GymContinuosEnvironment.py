import gym
from .Environment import Environment

class GymContinuosEnvironment(Environment):
    def __init__(self,environment,render):
        self.env = gym.make(environment)
        #self.num_agents = num_agents
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.world_status = False
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.step_count = 0
        if render:
            self.load_viewer()


    def step(self,action):
            self.acs = action

            # if there's a discrete space or more than one action this might be necessary
            # why is there an argmax here?
            # action = np.argmax(action)
            # print(action)
            # input()
            self.obs,self.rews,self.world_status, info = self.env.step(action)
            self.step_count +=1 

            return self.obs,self.rews,self.world_status

    def reset(self):
        self.obs = self.env.reset()

    def set_observation_space(self):
        observation_space = 1
        if self.env.observation_space.shape != ():
            for i in range(len(self.env.observation_space.shape)):
                observation_space *= self.env.observation_space.shape[i]
        else:
            observation_space = self.env.observation_space.n

        return observation_space

    def set_action_space(self):
        action_space = 1
        if self.env.action_space.shape != ():
            for i in range(len(self.env.action_space.shape)):
                action_space *= self.env.action_space.shape[i]
        else:
            action_space = self.env.action_space.n

        return action_space

    def get_observation(self):
        return self.obs

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        self.env.render()
import gym
import numpy as np
from .EvaluationEnvironment import EvaluationEnvironment

class GymDiscreteEvaluationEnvironment(EvaluationEnvironment):

    def __init__(self, configs):
        super(GymDiscreteEvaluationEnvironment,self).__init__(configs)
        self.env = gym.make(self.configs['env_name'])
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.render = configs['env_render']
        self.world_status = False

    def step(self, action):
        self.acs = action
        self.obs, self.rews, self.world_status, info = self.env.step(np.argmax(action))
        self.step_count += 1
        self.load_viewer()


        return self.obs, [self.rews], self.world_status, {'raw_reward': [self.rews]}

    def reset(self):
        self.step_count = 0
        self.obs = self.env.reset()

    def get_observation(self):
        return self.obs

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.rews


    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()

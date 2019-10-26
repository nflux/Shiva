import gym
import numpy as np
from .Environment import Environment

class GymDiscreteEnvironment(Environment):
    def __init__(self, configs):
        super(GymDiscreteEnvironment, self).__init__(configs)
        self.env = gym.make(self.env_name)
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.world_status = False
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.step_count = 0
        self.render = configs['env_render']

    def step(self, action):
        self.acs = action
        self.obs, self.rews, self.world_status, info = self.env.step(np.argmax(action))
        self.step_count += 1
        self.load_viewer()
        return self.obs, self.rews, self.world_status

    def reset(self):
        self.step_count = 0
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
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()

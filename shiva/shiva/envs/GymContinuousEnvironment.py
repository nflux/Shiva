import gym
from .Environment import Environment

class GymContinuousEnvironment(Environment):
    def __init__(self,environment):
        super(GymContinuousEnvironment,self).__init__(environment)
        self.env = gym.make(self.env_name)
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.world_status = False
        self.action_space_continuous = None
        self.action_space_discrete = None
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.step_count = 0

    def step(self, action, **kwargs):
            self.acs = action
            self.obs, self.rews, self.world_status, info = self.env.step(action)
            self.step_count +=1
            self.load_viewer()

            if self.normalize:
                return self.obs, self.normalize_reward(self.rews), self.world_status, {'raw_reward': self.rews, 'action':action}
            else:
                return self.obs, self.rews, self.world_status, {'raw_reward': self.rews, 'action':action}

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
            self.action_space_continuous = action_space
            self.action_space = action_space
        else:
            action_space = self.env.action_space.n
            self.action_space_discrete = action_space
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

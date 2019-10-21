import gym
import numpy as np

def initialize_env(env_params):

    if env_params['env_type'] == 'Gym':
        env = GymEnvironment(env_params['environment'], env_params['env_render'])

    return env


class AbstractEnvironment():
    def __init__(self, environment):
        self.step_count = 0

    def step(self,actions):
        pass

    def get_observation(self, agent):
        pass

    def get_observations(self):
        pass

    def get_action(self, agent):
        pass

    def get_actions(self):
        pass

    def get_reward(self, agent):
        pass

    def get_rewards(self):
        pass

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



class GymEnvironment(AbstractEnvironment):
    def __init__(self, environment, render=False):
        self.env_name = environment
        self.env = gym.make(environment)
        # self.num_agents = num_agents
        self.obs = self.env.reset()
        self.acs = 0
        self.rews = 0
        self.world_status = False
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.step_count = 0
        self.render = render

        # self.reset()

    def step(self,action):
        self.acs = action
        self.obs, self.rews, self.world_status, info = self.env.step(np.argmax(action))
        self.step_count += 1
        self.load_viewer()
        return self.obs, [self.rews], self.world_status

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
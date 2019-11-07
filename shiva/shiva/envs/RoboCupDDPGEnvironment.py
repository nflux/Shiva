import numpy as np
np.random.seed(5)
import torch
torch.manual_seed(5)
from .robocup.rc_env import rc_env
from .Environment import Environment

class RoboCupDDPGEnvironment(Environment):
    def __init__(self, config):
        super(RoboCupDDPGEnvironment, self).__init__(config)
        self.env = rc_env(config)
        self.env.launch()
        self.left_actions = self.env.left_actions
        self.left_params = self.env.left_action_params
        self.obs = self.env.left_obs
        self.rews = self.env.left_rewards
        self.world_status = self.env.world_status
        self.observation_space = self.env.left_features
        self.action_space = {'discrete': self.env.acs_dim, 'param': self.env.acs_param_dim}
        self.step_count = 0
        self.render = self.env.config['env_render']
        self.done = self.env.d

        self.load_viewer()

    def step(self, actions):
        # print('given actions', actions)

        self.left_actions = torch.tensor([np.argmax(actions[0:3])])

        self.left_params = torch.tensor([actions[3:]])
        
        self.obs, self.rews, _, _, self.done, _ = self.env.Step(left_actions=self.left_actions, left_params=self.left_params)

        # if self.rews[0] > 0.01:
        print('\nreward:', self.rews, '\n')
        return self.obs, self.rews, self.done, {'raw_reward': self.rews}

    def get_observation(self):
        return self.obs

    def get_actions(self):
        return self.left_actions, self.left_params

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env._start_viewer()

    def close(self):
        pass
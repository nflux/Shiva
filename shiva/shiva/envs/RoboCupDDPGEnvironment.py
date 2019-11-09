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

        # print(self.obs)
        # print(len(self.obs[0]))
        # input()

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

from pynput.keyboard import Key, KeyCode, Listener

class HumanPlayerInterface():
    '''
        Only for RoboCup
    '''
    def __init__(self):
        self.q = []
        self.listener = Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        # print('{0} release'.format(key))
        self.q.append(key)
        # if key == Key.esc:
        #     # Stop listener
        #     return False

    def get_action(self, obs):
        '''

        '''
        if len(self.q) > 0:
            action = self.q.pop(0)
            return self.robocup_action(action, obs)
        else:
            return None

    def robocup_action(self, action, obs):
        # print(action, type(action), self.q)
        # print(obs)
        # input()
        if action == Key.up:
            '''
                Rotate agent NORTH
            '''
            theta = 90
            action = [0, 1, 0, 0, 0, theta, 0, 0]
        elif action == Key.down:
            '''
                Rotate agent SOUTH
            '''
            theta = 90
            action = [0, 1, 0, 0, 0, theta, 0, 0]
        elif action == Key.right:
            '''
                Rotate agent EAST
            '''
            theta = 90
            action = [0, 1, 0, 0, 0, theta, 0, 0]
        elif action == Key.left:
            '''
                Rotate agent WEST
            '''
            theta = 90
            action = [0, 1, 0, 0, 0, theta, 0, 0]
        elif action == KeyCode.from_char('k'):
            '''
                Agent kicks
            '''
            kick_degree = 0
            kick_power = 50
            action = [0, 0, 1, 0, 0, 0, kick_power, kick_degree]
        elif action == KeyCode.from_char('d'):
            '''
                Agent dashes
            '''
            dash_degree = 0
            dash_power = 50
            action = [1, 0, 0, dash_power, dash_degree, 0, 0, 0]
        else:
            assert False, "Wrong action given"

        return np.array(action)
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
from math import atan2, pi, acos

class HumanPlayerInterface():
    '''
        Only for RoboCup
    '''
    def __init__(self):
        self.q = []
        self.listener = Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        self.q.append(key)

    def get_action(self, obs):
        '''

        '''
        if len(self.q) > 0:
            action = self.q.pop(0)
            return self.robocup_action(action, obs)
        else:
            return None

    '''
        Manual Control

            u - dash forward
            j - dash backward
            ; - 45 degree turn left
            ' - 45 degree turn right
        
    '''


    def robocup_action(self, action, obs):

        y_rad = obs[0,4]
        x_rad = obs[0,5]

        # print out observations of the ball (x y coordinates)
        # also print out the coordinates of the agent

        # test the x y coordinates to make sure the reward function is using those coordinates and not something else
        

        # check that if the agent can't see the ball whether or not the coordinates of the ball become -1; therefor invalid
        # so then we might need the true values if they are invalid
        # if they are invalid then we need to get the true true from the coach

        print("sin:", y_rad)
        print("cos", x_rad)

        # acos method for getting global angle
        th = acos(x_rad) * 180 / pi
        if y_rad < 0:
            th *= -1
        print("global angle(acos):", th)

        # arctan2 method for getting global angle
        global_theta = atan2(y_rad, x_rad) * 180 / pi
        print("global angle(atan2):", global_theta)
        
        if action == KeyCode.from_char('u'):

            print("Dash")

            dash_degree = 0
            dash_power = self.normalize_power(50)

            action = [1, 0, 0, dash_power, dash_degree, 0, 0, 0]


        elif action == KeyCode.from_char('j'):

            print("Dash")

            dash_degree = 0
            dash_power = self.normalize_power(-50)

            action = [1, 0, 0, dash_power, dash_degree, 0, 0, 0]

        elif action == KeyCode.from_char(';'):
            '''
                Turn Agent EAST
            '''

            print("East")

            action = [0, 1, 0, 0, 0, -0.25, 0, 0]

        elif action == KeyCode.from_char("'"):
            '''
                Turn Agent WEST
            '''

            print("West")

            action = [0, 1, 0, 0, 0, .25, 0, 0]

        elif action == KeyCode.from_char("q"):
            '''
                Agent Kick
            '''

            print("Kick")

            kick_degree = 0
            kick_power = self.normalize_power(50)

            action = [0, 0, 1, 0, 0, 0, kick_power, kick_degree]


        else:
            assert False, "Wrong action given"

        return np.array(action)

    def normalize_angle(self, delta):
        return 2 * (delta + 180) / 360

    def normalize_power(self, power):
        return (1-(-1)) * (power + 100) / 200
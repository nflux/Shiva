import torch
import numpy as np
from shiva.envs.robocup.rc_env import rc_env
from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot, action2one_hot_v
from torch.distributions import Categorical


class RoboCupEnvironment(Environment):
    def __init__(self, config, port=None):
        super(RoboCupEnvironment, self).__init__(config)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if port != None:
            self.env = rc_env(config, port)
            self.port = port
        else:
            self.env = rc_env(config, self.port)
        self.env.launch()

        self.left_actions = self.env.left_actions
        self.left_action_option = self.env.left_action_option

        self.obs = self.env.left_obs
        self.rews = self.env.left_rewards
        # self.world_status = self.env.world_status
        self.observation_space = self.env.left_features
        self.action_space = {
            'discrete': self.env.acs_dim,
            'continuous':0,
            'param': self.env.acs_dim,
            'acs_space': self.env.acs_dim 
        }
        self.step_count = 0
        self.render = self.env.env_render
        self.done = self.env.d

        self.load_viewer()

        # RoboCup specific metrics
        self.reward_per_episode = 0
        self.kicks = 0
        self.turns = 0
        self.dashes = 0
        self.goal_ctr = 0
    
    def isImit(self):
        return self.run_imit
    
    def isDescritized(self):
        return self.action_level == 'discretized'
    
    def descritize_action(self, action):
        return self.env.descritize_action(action)
    
    def isGoal(self):
        return self.env.checkGoal()

    def step(self, actions, discrete_select='sample', collect=True, device='cpu'):
        '''
            Input
                @actions
                @discrete_select        Specify how to choose the discrete action taken
                                            argmax      do an argmax on the discrete side
                                            sample      take the discrete side as a probability distribution to sample from

            Return
                A set with the following datas in order
                    next_observation
                    reward
                    done
                    more_data           list of [
                                            raw_reward,         useful when rewards are normalized
                                            action_taken        useful if discrete_select == 'sample'
                                        ]

        '''


        if discrete_select == 'argmax':
            if type(actions).__module__ == np.__name__:
                action = torch.from_numpy(actions)
            # if self.steps_per_episode == 0:
            print("argmax",actions)
            act_choice = torch.argmax(actions[:self.action_space['acs_space']])
        elif discrete_select == 'sample':
            # if self.steps_per_episode == 0:
            print("sample",actions)
            act_choice = Categorical(actions[:self.action_space['acs_space']]).sample()
        elif discrete_select == 'imit_discrete':
            act_choice = actions[0]
            # action = action2one_hot(self.acs_discrete, action.item())
            # act_choice = np.random.choice(self.action_space['discrete'], p=actions[:self.action_space['discrete']])

        # self.left_actions = act_choice.unsqueeze(dim=-1)
        self.steps_per_episode += 1
        if self.action_level == 'discretized':
            self.left_action_option[0] = act_choice
            
            # indicates whether its a dash, turn, or kick action from the action matrix
            if 0 <= self.left_action_option[0] < self.env.dash_idx:
                self.left_actions[0] = 0
                self.dashes += 1
            elif self.env.dash_idx <= self.left_action_option[0] < self.env.turn_idx:
                self.left_actions[0] = 1
                self.turns += 1
            else:
                self.left_actions[0] = 2
                self.kicks += 1

            # indicates whether its a dash, turn, or kick action from the action matrix
            # if 0 <= self.left_action_option[0] < self.env.dash_idx:
            #     self.left_actions[0] = 0
            #     self.dashes += 1
            # else:
            #     self.left_actions[0] = 1
            #     self.turns += 1

            # if self.left_action_option[0].item() < self.env.dash_idx:
            #     self.left_actions[0] = 0
            #     # print("DASH")
            #     self.dashes += 1
            # else:
            #     # print("KICK")
            #     self.left_actions[0] = 1
            #     self.kicks += 1

            # # Experiment 2
            # self.left_actions[0] = 0
            # self.kicks += 1

            self.obs, self.rews, _, _, self.done, _ = self.env.Step(left_actions=self.left_actions, left_options=self.left_action_option)
            # self.obs, self.rews, _, _, self.done, _ = self.env.Step(left_actions=torch.tensor([2]), left_options=torch.tensor([552]))
            actions_v = action2one_hot_v(self.action_space['acs_space'], act_choice)
            print(self.rews)
        else:
            self.left_actions = act_choice.unsqueeze(dim=0)
            self.left_action_option = actions[self.action_space['acs_space']:].unsqueeze(dim=0)
            # self.left_params = torch.tensor([params]).float()

            self.obs, self.rews, _, _, self.done, _ = self.env.Step(left_actions=self.left_actions, left_options=self.left_action_option)
            actions_v = torch.cat([action2one_hot_v(self.action_space['acs_space'], act_choice.item()).to(device), self.left_action_option[0]])

        if collect:
            self.collect_metrics()

        # print(actions_v)
        return self.obs, self.rews, self.done, {'raw_reward': self.rews, 'action': actions_v}

    def get_observation(self):
        return self.obs

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
    
    def get_imit_obs_msg(self):
        return self.env.getImitObsMsg()

    def get_actions(self):
        return self.left_actions, self.left_action_option

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env._start_viewer()

    def close(self):
        self.env.close = True
        self.env.d = True

    def is_done(self):
        return self.done

    def reset(self):
        self.kicks = 0
        self.turns = 0
        self.dashes = 0
        self.reward_per_episode = 0
        self.steps_per_episode = 0
        self.done = False
    
    def collect_metrics(self):
        '''
        Metrics collection
            Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
            Cumulative # of steps           self.step_count
            Cumulative # of episodes        self.done_count
            Episodic Reward                 self.reward_per_episode
            Goal Total                      self.goal_ctr
        '''
        self.steps_per_episode += 1
        self.step_count += 1
        self.done_count += 1 if self.done else 0
        self.reward_per_episode += self.rews[0]
        self.goal_ctr += 1 if self.isGoal() else 0

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.rews)
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Kicks_per_Episode', self.kicks),
                ('Turns_per_Episode', self.turns),
                ('Dashes_per_Episode', self.dashes),
                ('Agent/Steps_Per_Episode', self.steps_per_episode),
                ('Goal_Percentage/Per_Episodes', (self.goal_ctr/self.done_count)*100.0)
            ]

            print("Episode {} complete. Total Reward: {}".format(self.done_count, self.reward_per_episode))


        return metrics

from pynput.keyboard import Key, KeyCode, Listener
from math import atan2, pi, acos

class HumanPlayerInterface():

    # Only for RoboCup
    def __init__(self):
        self.q = []
        self.KEY_DASH = KeyCode.from_char('u')
        self.KEY_TURN_LEFT = KeyCode.from_char(';')
        self.KEY_TURN_RIGHT = KeyCode.from_char("'")
        self.KEY_KICK = KeyCode.from_char("q")

        self.listener = Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        if self.is_valid_key(key):
            self.q.append(key)

    def is_valid_key(self, key):
        return key in [self.KEY_DASH, self.KEY_TURN_LEFT, self.KEY_TURN_RIGHT, self.KEY_KICK]

    def get_action(self, obs):
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

        # obs.shape, 
        # print(obs[0, 17])
        print("(x,y):", obs[0,4], obs[0, 5])

        # print(obs[0, 15], '\n', obs[0, 16], '\n')

        # x_rad = obs[0,4]
        # y_rad = obs[0,5]

        # print out observations of the ball (x y coordinates)
        # also print out the coordinates of the agent

        # test the x y coordinates to make sure the reward function is using those coordinates and not something else


        # check that if the agent can't see the ball whether or not the coordinates of the ball become -1; therefor invalid
        # so then we might need the true values if they are invalid
        # if they are invalid then we need to get the true true from the coach

        # print("sin:", y_rad)
        # print("cos", x_rad)

        # acos method for getting global angle
        # th = acos(x_rad) * 180 / pi
        # if y_rad < 0:
        #     th *= -1
        # print("global angle(acos):", th)

        # arctan2 method for getting global angle
        # global_theta = atan2(y_rad, x_rad) * 180 / pi
        # print("global angle(atan2):", global_theta)

        # if self.action_level discretized

        # if action == self.KEY_DASH:
        #     '''
        #         Dash forward
        #     '''
        #     dash_degree = 0
        #     dash_power = self.normalize_power(50)

        #     action = [1, 0, 0, dash_power, dash_degree, 0, 0, 0]

        # elif action == self.KEY_TURN_LEFT:
        #     '''
        #         Turn Agent Left
        #     '''
        #     action = [0, 1, 0, 0, 0, -0.25, 0, 0]

        # elif action == self.KEY_TURN_RIGHT:
        #     '''
        #         Turn Agent Right
        #     '''
        #     action = [0, 1, 0, 0, 0, .25, 0, 0]

        # elif action == self.KEY_KICK:
        #     '''
        #         Agent Kick
        #     '''
        #     kick_degree = 0
        #     kick_power = self.normalize_power(50)

        #     action = [0, 0, 1, 0, 0, 0, kick_power, kick_degree]
        # else:
        #     assert False, "Wrong action given"



        if action == self.KEY_DASH:
            '''
                Dash forward
            '''
            dash_degree = 0
            dash_power = self.normalize_power(50)

            action = (10,0)
            action = 0

        elif action == self.KEY_TURN_LEFT:
            '''
                Turn Agent Left
            '''
            action = (22.5,)
            action = 1

        elif action == self.KEY_TURN_RIGHT:
            '''
                Turn Agent Right
            '''
            action = (-22.5,)
            action = 2

        elif action == elf.KEY_KICK:
            '''
                Agent Kick
            '''
            kick_degree = 0
            kick_power = self.normalize_power(50)

            action = (50,0)
            action = 3
        else:
            assert False, "Wrong action given"            

        return np.array(action)

    def normalize_angle(self, delta):
        return 2 * (delta + 180) / 360

    def normalize_power(self, power):
        return (1-(-1)) * (power + 100) / 200

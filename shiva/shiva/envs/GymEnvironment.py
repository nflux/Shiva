import gym
import numpy as np
from torch.distributions import Categorical
from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot
import torch

class GymEnvironment(Environment):
    def __init__(self, configs, *args, **kwargs):
        super(GymEnvironment, self).__init__(configs)
        self.env = gym.make(self.env_name)
        self.env.seed(self.manual_seed)
        np.random.seed(self.manual_seed)

        '''Set some attribute for Gym on MPI'''
        self.num_agents = 1
        self.roles = [self.env_name]
        self.num_instances_per_role = 1
        self.num_instances_per_env = 1

        self.action_space = {self.roles[0]: self.get_gym_action_space()}
        self.observation_space = {self.roles[0]: self.get_gym_observation_space()}
        self.reset()

        self.temp_done_counter = 0

    def step(self, action, discrete_select='argmax'):
        if not torch.is_tensor(action):
            action = torch.tensor(action)
        self.acs = action

        if self.is_action_space_discrete():
            '''Discrete, argmax or sample from distribution'''
            if not torch.is_tensor(action):
                action = torch.from_numpy(action)
            if discrete_select == 'argmax':
                action4Gym = torch.argmax(action).item()
            elif discrete_select == 'sample':
                action4Gym = Categorical(action).sample().item()
            self.obs, self.reward_per_step, self.done, info = self.env.step(action4Gym)
        else:
            '''Continuous actions'''
            # self.obs, self.reward_per_step, self.done, info = self.env.step([action[action4Gym.item()]])
            self.obs, self.reward_per_step, self.done, info = self.env.step(action)

        '''If Observation Space is discrete, turn it into a one-hot encode'''
        self.obs = self.transform_observation_space(self.obs)

        self.rew = self.normalize_reward(self.reward_per_step) if self.normalize else self.reward_per_step

        self.load_viewer()
        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
                Cumulative # of steps           self.step_count
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total
        '''
        self.steps_per_episode += 1
        self.step_count += 1
        self.done_count += 1 if self.done else 0
        self.reward_per_episode += self.rew
        self.reward_total += self.rew

        return self.obs, self.rew, self.done, {'raw_reward': self.reward_per_step, 'action': action}

    def reset(self, *args, **kwargs):
        self.steps_per_episode = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.done = False
        self.obs = self.transform_observation_space(self.env.reset())

    def get_metrics(self, episodic):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step),
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return [metrics] # single role metrics!
        # return metrics

    def is_done(self):
        return self.done

    def transform_observation_space(self, raw_obs):
        if self.is_observation_space_discrete():
            _one_hot_obs = np.zeros(self.observation_space)
            _one_hot_obs[raw_obs] = 1
            return _one_hot_obs
        else:
            return raw_obs

    def is_observation_space_discrete(self):
        return self.env.observation_space.shape == ()

    def get_gym_observation_space(self):
        observation_space = 1
        # if self.env.observation_space.shape != ():
        if not self.is_observation_space_discrete():
            for i in range(len(self.env.observation_space.shape)):
                observation_space *= self.env.observation_space.shape[i]
        else:
            observation_space = self.env.observation_space.n
        return observation_space

    def is_action_space_discrete(self):
        return self.env.action_space.shape == ()

    def get_gym_action_space(self):
        if self.is_action_space_discrete():
            return {
                'discrete': (self.env.action_space.n,),
                'continuous': 0,
                'param': 0,
                'acs_space': (self.env.action_space.n,),
                'actions_range': []
            }
        else:
            return {
                'discrete': 0,
                'continuous': self.env.action_space.shape,
                'param': 0,
                'acs_space': self.env.action_space.shape,
                'actions_range': [self.env.action_space.low, self.env.action_space.high]
            }

    def get_observations(self):
        return self.obs

    def get_observation(self):
        return self.obs

    def get_actions(self):
        return self.acs

    def get_action(self):
        return self.acs

    def get_reward(self):
        return torch.tensor(self.reward_per_step)

    def get_total_reward(self):
        '''
            Returns episodic reward
        '''
        return self.reward_per_episode

    def get_reward_episode(self, roles=False):
        if roles:
            return {self.roles[0]:self.reward_per_episode}
        return self.reward_per_episode

    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        self.env.close()

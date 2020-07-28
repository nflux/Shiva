import torch
import numpy as np
from shiva.core.admin import logger

class Environment:
    observations = []
    actions = []
    rewards = []
    dones = []

    steps_per_episode = 0
    step_count = 0
    done_count = 0
    reward_per_step = 0
    reward_per_episode = 0
    reward_total = 0

    total_episodes_to_play = None

    def __init__(self, configs):
        {setattr(self, k, v) for k,v in configs['Environment'].items()}
        self.configs = configs

        # for previous versions support on attribute names
        if not hasattr(self, 'num_instances') and not hasattr(self, 'num_envs'):
            self.num_envs = 1
        if hasattr(self, 'num_instances') and not hasattr(self, 'num_envs'):
            self.num_envs = self.num_instances

        # normalization factors
        self.reward_factor = self.reward_factor if hasattr(self, 'reward_factor') else 1
        self.max_reward = self.max_reward if hasattr(self, 'max_reward') else 1
        self.min_reward = self.min_reward if hasattr(self, 'min_reward') else -1

    def step(self,actions):
        pass

    def finished(self, n_episodes=None):
        '''
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        '''
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def start_env(self):
        return True

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

    def get_metrics(self):
        '''
            To be implemented per Environment
        '''
        pass

    def reset(self):
        pass

    def load_viewer(self):
        pass

    def normalize_reward(self, reward):
        return self.reward_factor*(reward-self.min_reward)/(self.max_reward-self.min_reward)

    def _normalize_reward(self, reward):
        return (self.b-self.a)*(reward-self.min)/(self.max-self.min)

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Env']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<{}>".format(self.__class__.__name__)

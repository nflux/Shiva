import torch
from typing import Dict, List, Tuple, Any, Union
import numpy as np
from shiva.core.admin import logger
from abc import abstractmethod


class Environment:
    """
    Abstract Environment Class all environments implemented in Shiva inherit from.
    """
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

    def __init__(self, configs: Dict):
        {setattr(self, k, v) for k, v in configs['Environment'].items()}
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

    @abstractmethod
    def step(self, actions) -> None:
        """
        Passes in an action which is in turn passed into the environment to continue the 
        reinforcement training loop.
        """

    def finished(self, n_episodes=None) -> bool:
        '''
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        '''
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def start_env(self) -> bool:
        """

        """
        return True

    @abstractmethod
    def get_observation(self, agent) -> None:
        """
        Returns an observation.
        """

    @abstractmethod
    def get_observations(self) -> None:
        """
        Returns multiple observations.
        """

    @abstractmethod
    def get_action(self, agent):
        """
        Args:
            agent (Agent): 
                The agent from which you want to get the action from.
        Returns an action.
        """

    @abstractmethod
    def get_actions(self)-> None:
        """
        Returns multiple actions.
        """
    @abstractmethod
    def get_reward(self, agent: object) -> None:
        """
        
        """

    @abstractmethod
    def get_rewards(self) -> None:
        """
        
        """

    def get_observation_space(self) -> np.array:
        """
        This could return either a scalar, list, array, or tensor depending upon implementation.
        """
        return self.observation_space

    def get_action_space(self) -> np.array:
        return self.action_space

    def get_current_step(self) -> int:
        """
        Returns:
            Current step in the enpisode.
        """
        return self.step_count

    @abstractmethod
    def get_metrics(self) -> None:
        '''
        To be implemented per Environment
        '''

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the environment.
        """

    @abstractmethod
    def load_viewer(self) -> None:
        """
        Loads the environment's renderer.
        """

    def normalize_reward(self, reward) -> float:
        """
        
        """
        return self.reward_factor*(reward-self.min_reward)/(self.max_reward-self.min_reward)

    def _normalize_reward(self, reward) -> float:
        """
        
        """
        return (self.b-self.a)*(reward-self.min)/(self.max-self.min)

    def log(self, msg, to_print=False, verbose_level=-1) -> None:
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Env']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self) -> str:
        """
        
        """
        return "<{}>".format(self.__class__.__name__)

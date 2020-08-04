import torch
from typing import Dict, List, Tuple, Any, Union
import numpy as np
from shiva.core.admin import logger
from abc import abstractmethod, ABC


class Environment(ABC):
    """ Abstract Environment Class all environments implemented in Shiva inherit from.
    """

    def __init__(self, configs: Dict):
        {setattr(self, k, v) for k, v in configs['Environment'].items()}
        self.configs = configs

        # Environment parameters
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.steps_per_episode = 0
        self.step_count = 0
        self.done_count = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.reward_total = 0

        self.total_episodes_to_play = None


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
        """
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        """
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def start_env(self) -> bool:
        """ Typically can be used to launch the environment and set some initial values."""
        return True

    @abstractmethod
    def get_observation(self, agent) -> None:
        """Returns an observation."""

    @abstractmethod
    def get_observations(self) -> None:
        """Returns multiple observations."""

    @abstractmethod
    def get_action(self) -> None:
        """ Returns an action."""

    @abstractmethod
    def get_actions(self) -> None:
        """Returns multiple actions."""
    @abstractmethod
    def get_reward(self) -> None:
        """Returns the reward stepwise reward."""

    @abstractmethod
    def get_rewards(self) -> None:
        """Returns stepwise reward for all the agents."""

    def get_observation_space(self) -> np.array:
        """Returns the observation space dimension for the environment.
        Returns:
            Tuple of integers representing observation dimensions.
        """
        return self.observation_space

    def get_action_space(self) -> Tuple:
        """Returns the action space dimensions for the environment.
        Returns:
            Tuple of integers representing the actions dimensions.
        """
        return self.action_space

    def get_current_step(self) -> int:
        """ Returns the current step in the episode.
        Returns:
            Current step in the episode.
        """
        return self.step_count

    @abstractmethod
    def get_metrics(self) -> None:
        """Returns the metrics relevant to the environment.
        Returns:
            None
        """

    @abstractmethod
    def reset(self) -> None:
        """Resets the environment state and parameters.
        Returns:
            None
        """

    @abstractmethod
    def load_viewer(self) -> None:
        """Loads the environment's renderer.
        Returns:
            None
        """

    def normalize_reward(self, reward) -> float:
        """ Used to normalize the reward and multiply it by a factor.

        Experimental.

        Returns:
            Normalized reward multiplied by some factor given in the config.
        """
        return self.reward_factor*(reward-self.min_reward)/(self.max_reward-self.min_reward)

    def _normalize_reward(self, reward) -> float:
        """ Used to normalize the reward to a specified interval.

        Returns:
            Normalized reward to a given interval.
        """
        return (self.b-self.a)*(reward-self.min)/(self.max-self.min)

    def log(self, msg, verbose_level=-1) -> None:
        """If verbose_level is not given, by default will log

        Used to output debug messages.
        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['Env']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, self.configs['Admin']['print_debug'])

    def __str__(self) -> str:
        """Useful for outputting the class name.

        Returns (str):
            Class name as a string.
        """
        return "<{}>".format(self.__class__.__name__)

import torch
import numpy as np
from shiva.core.admin import logger
from shiva.helpers.misc import set_seed

class Environment:
    """ Abstract Environment Class all environments implemented in Shiva inherit from.
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

    action_selection_mode = 'argmax'

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

        set_seed(self.manual_seed)
        self.log(f"MANUAL SEED {self.manual_seed}")


    def step(self, actions):
        """
        Passes in an action which is in turn passed into the environment to continue the
        reinforcement training loop.
        """
        pass

    def finished(self, n_episodes=None):
        """
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        """
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def start_env(self):
        """ Typically can be used to launch the environment and set some initial values."""
        return True

    def get_observation(self, agent):
        """Returns an observation."""
        pass

    def get_observations(self):
        """Returns multiple observations."""
        pass

    def get_action(self, agent):
        """ Returns an action."""
        pass

    def get_actions(self):
        """Returns multiple actions."""
        pass

    def get_reward(self, agent):
        """Returns the reward stepwise reward."""
        pass

    def get_rewards(self):
        """Returns stepwise reward for all the agents."""
        pass

    def get_observation_space(self):
        """Returns the observation space dimension for the environment.
        Returns:
            Tuple of integers representing observation dimensions.
        """
        return self.observation_space

    def get_action_space(self):
        """Returns the action space dimensions for the environment.
        Returns:
            Tuple of integers representing the actions dimensions.
        """
        return self.action_space

    def get_current_step(self):
        """ Returns the current step in the episode.
        Returns:
            Current step in the episode.
        """
        return self.step_count

    def get_metrics(self):
        """Returns the metrics relevant to the environment.
        Returns:
            None
        """
        pass

    def action_selection(self, action) -> float:
        """Function performs the action selection at the Environment level determined by the config parameter 'action_selection_mode'
        Options for 'action_selection_mode' are 'argmax' or 'sample'

        Returns:
            float value of the action chosen
        """
        if self.action_selection_mode == 'argmax':
            return np.argmax(action).item()
        elif self.action_selection_mode == 'sample':
            return torch.distributions.Categorical(torch.from_numpy(action)).sample().item()
        else:
            assert False, f"Invalid 'action_selection_mode', got {self.action_selection_mode}. Only 'argmax' or 'sample' as implemented."

    def reset(self):
        """Resets the environment state and parameters.
        Returns:
            None
        """
        pass

    def load_viewer(self):
        """Loads the environment's renderer.
        Returns:
            None
        """
        pass

    def normalize_reward(self, reward):
        """ Used to normalize the reward and multiply it by a factor.

        Experimental.

        Returns:
            Normalized reward multiplied by some factor given in the config.
        """
        return self.reward_factor*(reward-self.min_reward)/(self.max_reward-self.min_reward)

    def _normalize_reward(self, reward):
        """ Used to normalize the reward to a specified interval.

        Returns:
            Normalized reward to a given interval.
        """
        return (self.b-self.a)*(reward-self.min)/(self.max-self.min)

    def log(self, msg, to_print=False, verbose_level=-1):
        """If verbose_level is not given, by default will log

        Used to output debug messages.
        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['Env']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        """Useful for outputting the class name.

        Returns (str):
            Class name as a string.
        """
        return "<{}>".format(self.__class__.__name__)

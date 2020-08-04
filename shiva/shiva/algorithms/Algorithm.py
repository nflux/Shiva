import torch
from shiva.core.admin import logger
from shiva.agents import Agent

from typing import List, Dict, Tuple, Any, Union


class Algorithm:
    def __init__(self, observation_space: Dict[str, int], action_space: Dict[str, Dict[str, Tuple[Union[int]]]], configs: Dict[str, Any]) -> None:
        """
        Abstract class for the algorithm. Several values are initialized.

        Args:
            observation_space (Dict[str, int]):
            action_space (Dict[str, Dict[str, Tuple[int, ...]]]): This is the action space dictionary that our Environment wrappers output (link to Environment)
            configs (Dict[str, ...]): The global config used for the run
        """
        self.configs = configs
        {setattr(self, k, v) for k, v in self.configs['Algorithm'].items()}
        self.agentCount = 0
        self.agents = []
        self.observation_space = observation_space
        self.action_space = action_space
        self.loss_calc = getattr(torch.nn, self.configs['Algorithm']['loss_function'])()
        self.num_updates = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.metrics = []

    def update(self, agent, data, episodic=False):
        '''
            Updates the agents network using the data

            Input
                agent:      the agent who we want to update it's network
                data:       data used to train the network
                episodic:   flag indicating if the update is episodic or per timestep

            Return
                None
        '''
        assert "Method Not Implemented"

    def get_num_updates(self) -> int:
        return self.num_updates

    def create_agent(self):
        '''
            Creates a new agent

            Input

            Return
                Agent
        '''
        raise NotImplementedError("Method to be implemented by subclass")

    def save_central_critic(self, agent: Agent) -> None:
        """

        Args:
            agent:

        Returns:

        """
        pass

    def load_central_critic(self, agent):
        pass

    def get_agents(self):
        return self.agents

    def log(self, msg, to_print=False, verbose_level=-1) -> None:
        """

        Args:
            msg:
            to_print:
            verbose_level:

        Returns:

        """
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Algorithm']:
            text = '{}\t{}'.format(self, msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])
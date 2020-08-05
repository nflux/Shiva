import torch
from abc import abstractmethod

from shiva.core.admin import logger
from shiva.agents.Agent import Agent
from shiva.buffers.ReplayBuffer import ReplayBuffer

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

    @abstractmethod
    def update(self, agents: List[Agent], buffer: ReplayBuffer, *args, **kwargs) -> None:
        '''
        Updates the agents network using the data

        Args
            agents (List[Agent]): the agent who we want to update it's network
            data (ReplayBuffer): buffer containing the data used to train the network
            episodic: flag indicating if the update is episodic or per timestep

        Return
            None
        '''
        raise NotImplementedError("Method to be implemented by subclass")

    @abstractmethod
    def create_agent(self) -> Agent:
        '''
        Creates a new agent

        Return
            Agent
        '''
        raise NotImplementedError("Method to be implemented by subclass")

    def get_num_updates(self) -> int:
        """

        Returns:
            int: number of updates performed so far

        """
        return self.num_updates

    def save_central_critic(self, agent: Agent) -> None:
        """
        When using a central critic algorithm, we use this function to save the critic into the agents so that they are properly saved in file system.
        If algorithm is not central critic, default behaviour of this function is to pass.

        Args:
            agent (Agent): agent where we are going to store the critic network

        Returns:
            None
        """
        pass

    def load_central_critic(self, agent: Agent):
        """
        When using a central critic algorithm, we use this function to load the central critic from the agent into the central critic network.
        If algorithm is not central critic, default behaviour of this function is to pass.

        Args:
            agent (Agent): agent where we are going to load the critic network

        Returns:
            None
        """
        pass

    def get_agents(self) -> List[Agent]:
        """
        Get method that returns a list of Agents for which the Algorithm has currently pointers to.

        Returns:
            List[Agents]
        """
        return self.agents

    def log(self, msg, verbose_level=-1) -> None:
        """
        Logging function. Uses python logger and can optionally output to terminal depending on the config `['Admin']['print_debug']`

        Args:
            msg: Message to be logged
            verbose_level: verbose level used for the given message. Defaults to -1.

        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['Algorithm']:
            text = '{}\t{}'.format(self, msg)
            logger.info(text, self.configs['Admin']['print_debug'])
import torch
import numpy as np
from shiva.core.admin import logger

from typing import Dict, Tuple, List, Union, Any


class Algorithm:
    def __init__(self, observation_space: Dict[str, int], action_space: Dict[str, Dict[str, Tuple[Union[int]]]], configs: Dict[str, Any]) -> None:
        """
        Abstract class for the algorithm. Several values are initialized.

        Args:
            observation_space (Dict[str, int]):
            action_space (Dict[str, Dict[str, Tuple[int, ...]]]): This is the action space dictionary that our Environment wrappers output (link to Environment)
            configs (Dict[str, ...]): The global config used for the run
        Returns:
            None
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

    def update(self, agents: List, buffer, *args, **kwargs) -> None:
        """Updates the agents network using the data

        Args:
            agents (List[Agent]): the agent who we want to update it's network
            buffer (ReplayBuffer): buffer containing trajectories to sample from
            data (ReplayBuffer): buffer containing the data used to train the network
            episodic: flag indicating if the update is episodic or per timestep

        Returns:
            None
        """
        raise NotImplementedError("Method to be implemented by subclass")

    def create_agent(self):
        """Creates a new agent

        Returns:
            Agent
        """
        raise NotImplementedError("Method to be implemented by subclass")

    def get_num_updates(self) -> int:
        """
        Returns:
            int: number of updates performed so far
        """
        return self.num_updates

    def save_central_critic(self, agent) -> None:
        """
        When using a central critic algorithm, we use this function to save the critic into the agents so that they are properly saved in file system.
        If algorithm is not central critic, default behaviour of this function is to pass.

        Args:
            agent (Agent): agent where we are going to store the critic network

        Returns:
            None
        """
        pass

    def load_central_critic(self, agent):
        """
        When using a central critic algorithm, we use this function to load the central critic from the agent into the central critic network.
        If algorithm is not central critic, default behaviour of this function is to pass.

        Args:
            agent (Agent): agent where we are going to load the critic network

        Returns:
            None
        """
        pass

    def get_agents(self) -> List:
        """
        Get method that returns a list of Agents for which the Algorithm has currently pointers to.

        Returns:
            List[Agents]
        """
        return self.agents

    def id_generator(self) -> int:
        """ Generates a new id for an agent.
        Returns:
            Integer agent id.
        """
        agent_id = self.agentCount
        self.agentCount += 1
        return agent_id

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

import torch
import numpy as np
from shiva.core.admin import logger

class Algorithm():
    def __init__(self, obs_space, acs_space, configs):
        '''
            Input
                observation_space   Shape of the observation space, aka input to policy network
                action_space        Shape of the action space, aka output from policy network
                loss_function       Function used to calculate the loss during training
                regularizer
                recurrence
                optimizer           Optimization function to train network weights
                gamma               Hyperparameter
                learning_rate       Learning rate used in the optimizer
                beta                Hyperparameter
        '''
        self.configs = configs
        {setattr(self, k, v) for k, v in self.configs['Algorithm'].items()}
        self.agentCount = 0
        self.agents = []
        self.observation_space = obs_space
        self.action_space = acs_space
        if self.configs['Buffer']['prioritized']:
            self.loss_calc = getattr(torch.nn, self.configs['Algorithm']['loss_function'])(reduction='none')
        else:
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

    def evolve(self):
        raise NotImplemented

    def get_num_updates(self):
        return self.num_updates

    def get_action(self, agent, observation):
        '''
            Determines the best action for the agent and a given observation

            Input
                agent:          the agent we want the action
                observation:

            Return
                Action
        '''
        assert "Method Not Implemented"

    def create_agent(self):
        '''
            Creates a new agent

            Input

            Return
                Agent
        '''
        assert "Method Not Implemented"

    def save_central_critic(self, agent):
        pass

    def load_central_critic(self, agent):
        pass

    def id_generator(self):
        agent_id = self.agentCount
        self.agentCount += 1
        return agent_id

    def get_agents(self):
        return self.agents

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Algorithm']:
            text = '{}\t{}'.format(self, msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])
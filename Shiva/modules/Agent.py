import numpy as np
import torch
import os
import copy
import Network

class Agent:
    def __init__(self, id, obs_dim, action_dim, optimizer_function, learning_rate, config: dict):
        '''
        Base Attributes of Agent
            id = given by the learner
            obs_dim
            act_dim
            policy = Neural Network Policy
            target_policy = Target Neural Network Policy
            optimizer = Optimier Function
            learning_rate = Learning Rate
        '''
        self.id = id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy = None
        self.optimizer_function = optimizer_function
        self.learning_rate = learning_rate
        self.config = config

    def save(self, save_path, step):
        torch.save(self.policy, save_path + '/policy.pth')

    def load(self, load_path):
        torch.load(self.policy, load_path)

    def delete(self, id):
        path = os.getcwd()
        file_path = "{}/Shiva/ShivaAgent/{}".format(path,self.id)
        if os.path.exists(file_path):
            os.remove(file_path)

class DQAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DQAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)
        
    def get_action(self, obs):
        '''
            This should iterate over all the possible actions
        '''
        return self.policy(obs)

class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DDPGAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)


class ImitationAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(ImitationLearnerAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)

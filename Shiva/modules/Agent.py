import numpy as np
import torch
import os
import copy
import Network
import helpers

class Agent(object):
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def save(self, save_path, step):
        torch.save(self.policy, save_path + '/policy.pth')

    def load_net(self, load_path):
        self.policy = torch.load(load_path)

    def find_best_action(self, network, observation) -> np.ndarray:
        '''
            Iterates over the action space to find the one with the highest Q value

            Input
                network         policy network to be used
                observation     observation from the environment
            
            Returns
                A one-hot encoded list
        '''
        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.action_dim).to(self.device)
        for i in range(self.action_dim):
            act_v = helpers.action2one_hot_v(self.action_dim, i)
            q_val = network(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

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
            This method iterates over all the possible actions to find the one with the highest Q value
        '''
        return self.find_best_action(self.policy, obs)

    def get_action_target(self, obs):
        '''
            Same as above but using the Target network
        '''
        return self.find_best_action(self.target_policy, obs)


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




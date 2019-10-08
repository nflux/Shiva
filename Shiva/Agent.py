import numpy as np
import torch
import os
import uuid 
import copy

import Network

class Agent:
    def __init__(self, network_input, network_output, optimizer, learning_rate, config: dict):
        '''
        Base Attributes of Agent
            network_input
            network_output
            id = id Dimensions
            policy = Neural Network Policy
            target_policy = Target Neural Network Policy
            optimizer = Optimier Function
            learning_rate = Learning Rate

        '''
        # self.obs_dim = obs_dim
        # self.action_dim = action_dim
        self.network_input = network_input
        self.network_output = network_output
        self.id = uuid.uuid4()
        self.policy = None
        self.optimizer = None
        self.learning_rate = learning_rate
        self.config = config

    def save(self):
        '''
        # Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        pass  

    def load(self):
        '''
        # Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        pass


class DQAgent(Agent):
    def __init__(self, network_input, network_output, optimizer, learning_rate, config: dict):
        # Calls the Super Class Agent to do some initialization
        super(DQAgent,self).__init__(network_input, network_output, optimizer, learning_rate, config)
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        # Calls the optimizer for the policy
        self.optimizer = optimizer(params=self.policy.parameters(), lr=learning_rate)
    
    def save(self):
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        if os.path.exists("/ShivaAgent"+str(self.id)+ ".pth"):
             #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
            torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        else:
            print("The Load File for the Shiva Agent Model Does Not Exist")


       

'''
The Scenario when DQAgent is passed as a config_tuple
class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, config_tuple):Policy and Target Polict calls the dqnet. Hidden Layer 1 = 32 Hidden Layer 2= 6

        # Calls the Super Class Agent to do some initialization
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate)

        # Grabs the Tuples and split it into the proper variables, Optimizer, Learning_Rate, HIDDEN_LAYER 
        self.optimizer= config_tuple[0]
        self.learning_rate = config_tuple[1]
        HIDDEN_LAYER = config_tuple[2]

        # NOTE: The Network Policy should be changed, to make it more dynamic. 
        self.policy = dqnet(obs_dim, HIDDEN_LAYER[0], HIDDEN_LAYER[1], action_dim)
        self.target_policy = dqnet(obs_dim, HIDDEN_LAYER[0], HIDDEN_LAYER[1], action_dim)

        # Calls the optimizer for the policy
        self.optimizer = optimizer(params=self.policy.parameters(), lr=learning_rate)
    
    def save(self):
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

'''
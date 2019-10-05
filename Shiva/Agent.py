import numpy as np
import torch
# from Network import DQNet as dqnet
import Network_builder
from importlib import import_module
import uuid 
class Agent:
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config:list):
        '''
        Base Attributes of Agent
        obs_dim = Observation Dimensions
        action_dim = Action Dimensions
        id = id Dimensions
        policy = Neural Network Policy
        target_policy = Target Neural Network Policy
        optimizer = Optimier Function
        learning_rate = Learning Rate

        '''
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config:list):
        # Calls the Super Class Agent to do some initialization
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, config)

        # Policy and Target Polict calls the dqnet. Hidden Layer 1 = 32 Hidden Layer 2= 64 

        nb = Network_builder.NetworkBuilder(obs_dim + action_dim,1, config[1])
        network_name = nb.getFileName()
        dqnet = import_module (network_name)
        #self.policy = dqnet(obs_dim, 32,64,action_dim)
        #self.target_policy = dqnet(obs_dim, 32,64,action_dim)
        
        self.policy = dqnet.DQNet(obs_dim,action_dim)
        self.target_policy = dqnet.DQNet(obs_dim,action_dim)
        # Calls the optimizer for the policy
        self.optimizer = optimizer(params=self.policy.parameters(), lr=learning_rate)
    
    def save(self):
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

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
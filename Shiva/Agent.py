import numpy as np
import torch
import os
import uuid 
import copy

import Network

class Agent:
    def __init__(self, obs_dim, action_dim, optimizer_function, learning_rate, id, config: dict):
        '''
        Base Attributes of Agent
            obs_dim
            act_dim
            id = id
            policy = Neural Network Policy
            target_policy = Target Neural Network Policy
            optimizer = Optimier Function
            learning_rate = Learning Rate

        '''
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = id
        self.policy = None
        self.optimizer_function = optimizer_function
        self.learning_rate = learning_rate
        self.config = config

    def save(self, step):
        '''
        
        # Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        path = os.getcwd()
        directory = path+"/Shiva/ShivaAgent/"+str(self.id)+"/"
        if not os.path.exists(directory): 
            os.makedirs(directory)
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,path+"/Shiva/ShivaAgent/"+str(self.id)+"/"+str(step)+ ".pth")

    def load(self, step):
        '''
        # Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        path = os.getcwd()
        if os.path.exists(path+"/Shiva/ShivaAgent/"+str(self.id)+"/"+str(step)+ ".pth"):
            #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
            torch.load(self.policy,path+"/Shiva/ShivaAgent/"+str(self.id)+"/"+str(step)+ ".pth")
        else:
            print("The Load File for the Shiva Agent Model Does Not Exist")


class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, config: dict):
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate,id, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)



class DDPGAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, config: dict):
        super(DDPGAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate,id, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)


class ImitationLearnerAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, config: dict):
        super(ImitationLearnerAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate,id, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)



# NO Longer In Use Feel Free to Delete if Not needed. 
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


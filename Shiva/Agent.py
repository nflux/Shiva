import numpy as np
import torch
import os
import uuid 
import copy

import Network

class Agent:
    def __init__(self, obs_dim, action_dim, optimizer_function, learning_rate, config: dict):
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
        self.id = uuid.uuid4()
        self.policy = None
        self.optimizer_function = optimizer_function
        self.learning_rate = learning_rate
        self.config = config

    def save(self):
        '''
        # Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        '''
        # Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        if os.path.exists("/ShivaAgent"+str(self.id)+ ".pth"):
             #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
            torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        else:
            print("The Load File for the Shiva Agent Model Does Not Exist")


class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)
       

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

class DDPGAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        # Calls the Super Class Agent to do some initialization
        super(DDPGAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, config)
        
        actor_params = {
            'layers': config['network']['actor_layers'],
            'activation_function': config['network']['actor_activation_function'],
            'output_function': config['network']['actor_output_function']
        }
        self.actor = Network.initialize_network(obs_dim, action_dim, actor_params)
        self.actor_target = copy.deepcopy(self.actor)

        critic_head_params = {
            'layers': config['network']['critic_head_layers'],
            'activation_function': config['network']['critic_head_activation_function'],
            'output_function': config['network']['critic_head_output_function']
        }
        self.critic_head = Network.initialize_network(obs_dim, action_dim, critic_head_params)
        self.critic_head_target = copy.deepcopy(self.critic_head)
        
        critic_tail_params = {
            'layers': config['network']['critic_tail_layers'],
            'activation_function': config['network']['critic_tail_activation_function'],
            'output_function': config['network']['critic_tail_output_function']
        }
        self.critic_tail = Network.initialize_network(obs_dim, action_dim, critic_tail_params)
        self.critic_tail_target = copy.deepcopy(self.critic_tail)

        # Calls the optimizer for the policy
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)

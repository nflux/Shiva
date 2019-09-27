import numpy as np
import torch
from Network import DQNet as dqnet
import uuid 
class Agent:
    def __init__(self, obs_dim, action_dim, optimizer, learningrate):
        '''
        Base Attributes of Agent
        obs_dim = Observation Dimensions
        action_dim = Action Dimensions
        id = id Dimensions
        policy = Neural Network Policy
        target_policy = Target Neural Network Policy
        optimizer = Optimier Function
        learningrate = Learning Rate

        '''
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.id = uuid.uuid4()
        self.policy = None
        self.target_policy = None
        self.optimizer = None
        self.learningrate = learningrate

    def save(self):
        '''
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        pass  

    def load(self):
        '''
        #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''
        pass

class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learningrate):
        # Calls the Super Class Agent to do some initialization
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learningrate)

        # Policy and Target Polict calls the dqnet. Hidden Layer 1 = 32 Hidden Layer 2= 64 
        self.policy = dqnet(action_dim,32,64,obs_dim)

        self.target_policy = dqnet(action_dim,32, 64, obs_dim)
        # Calls the optimizer for the policy
        self.optimizer = optimizer(params=self.policy.parameters(), lr=learningrate)
    
    def save(self):
        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")

    def load(self):
        #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
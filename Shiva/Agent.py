import numpy as np
import torch
import os
import copy
import Network

class Agent:
    def __init__(self, obs_dim, action_dim, optimizer_function, learning_rate, id, root, config: dict):
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
        self.root = root

    def save(self, step):
        '''
        # Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''

        directory = self.root + "/Agents/" + str(self.id)

        if not os.path.exists(directory): 
            os.makedirs(directory)

        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy,directory + '/' + str(self.id) + 'Agent'  + "_" + str(step) + ".pth")

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

    def delete(self, id):
        path = os.getcwd()
        if os.path.exists(path+"/Shiva/ShivaAgent/"+str(self.id)):
            os.remove(path+"/Shiva/ShivaAgent/"+str(self.id))

class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, root, config: dict):
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, id, root, config)
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
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, root, config: dict):
        super(DDPGAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, id, root, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)


class ImitationAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, root, config: dict):
        super(ImitationLearnerAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, id, root, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)




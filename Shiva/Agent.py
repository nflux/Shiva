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

        directory = "{}/Agents/{}".format(self.root, self.id)

        if not os.path.exists(directory):
            os.makedirs(directory)

        save_path = "{}/{}Agent_{}.pth".format(directory, self.id, step)

        #Saves the current Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.save(self.policy, save_path)

    def load(self, path, configs):

        '''
        # Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
        torch.load(self.policy,"/ShivaAgent"+str(self.id)+ ".pth")
        '''

        # Checks if the agent has the right file extension
        assert path[:-4] == '.pth', "Wrong file extension!"

        cwd = os.getcwd()

        load_path = "{}/EliteAgents{}".format(cwd,path)

        if os.path.exists(load_path):

            if 'DQAgent' in path:
                return DQAgent(self.obs_dim,self.action_dim, configs['Algorithm']['optimizer'], configs['Algorithm']['learning_rate'], 0, self.root, configs['Network'])

            #Loads a Neural Network into a .pth with a name of ShivaAgentxxxx.pth
            torch.load(self.policy,load_path)
        else:
            print("The Load File for the Shiva Agent Model Does Not Exist")

    def delete(self, id):
        path = os.getcwd()
        file_path = "{}/Shiva/ShivaAgent/{}".format(path,self.id)
        if os.path.exists(file_path):
            os.remove(file_path)

class DQAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, id, root, config: dict):
        super(DQAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, id, root, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)



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
        super(ImitationAgent,self).__init__(obs_dim, action_dim, optimizer, learning_rate, id, root, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)

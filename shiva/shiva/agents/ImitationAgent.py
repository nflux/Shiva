from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
import copy
import torch.optim
import numpy as np

class ImitationAgent(Agent):
    def __init__(self, id, obs_dim, acs_discrete, acs_continuous, agent_config,net_config):
        self.acs_discrete = acs_discrete
        self.acs_continuous = acs_continuous
        if (acs_discrete != None) and (acs_continuous != None):
            self.network_output = self.acs_discrete + self.acs_continuous
        elif (self.acs_discrete == None):
            self.network_output = self.acs_continuous
        else:
            self.network_output = self.acs_discrete
        super(ImitationAgent,self).__init__(id, obs_dim, self.network_output, agent_config,net_config)
        self.id = id
        self.acs_discrete = acs_discrete
        self.acs_continuous = acs_continuous
        self.network_input = obs_dim
        self.policy = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config)
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.policy.parameters(), lr=agent_config['learning_rate'])


    def find_best_action(self,network,observation):
        full_action = network(torch.tensor(observation).float()).detach().numpy()
        discrete_action = full_action[:self.acs_discrete]

        if self.action_policy == 'argmax':
            discrete_action = np.random.choice(discrete_action)
        else:
            discrete_action = misc.action2one_hot(self.acs_space,np.argmax(discrete_action))

        full_action[:self.acs_discrete] = discrete_action
        return torch.from_numpy(full_action)

    '''def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:
        if self.action_policy =='argmax':
            return np.random.choice(network(torch.tensor(observation).float()).detach().numpy())
        else:
            return misc.action2one_hot(self.acs_space,np.argmax(network(torch.tensor(observation).float()).detach()).item())'''

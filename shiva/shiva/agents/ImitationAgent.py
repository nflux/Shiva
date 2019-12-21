import numpy as np
import copy
import torch.optim

from shiva.agents.Agent import Agent
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork as DLN
from shiva.helpers import misc

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
        self.network_input = obs_dim

        self.policy = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config)
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.policy.parameters(), lr=agent_config['learning_rate'])


    def find_best_action(self,network,observation):
        #retrieve the action given an observation
        full_action = network(torch.tensor(observation).float()).detach().numpy()
        if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))

        #Check if there is a discrete component to the action tensor
        if (self.acs_discrete != None):
            #If there is, extract the discrete component
            discrete_action = full_action[: self.acs_discrete]
            #Use argmax policy to obtain the discrete action to be taken
            if self.action_policy == 'argmax':
                discrete_action = misc.action2one_hot(self.acs_discrete,np.argmax(discrete_action)).tolist()
            #Sample from the discrete actions to determine which action to take
            else:
                sampled_choice = np.zeros(len(discrete_action))
                discrete_action = discrete_action.reshape((1,len(discrete_action)))
                sampled_action = np.random.choice(discrete_action)
                sampled_action = np.where(discrete_action == sampled_action)
                sampled_choice[sampled_action[0]] = 1.0
                discrete_action = sampled_choice.tolist()
            #Recombine the chosen discrete action with the continuous action values
            full_action = np.concatenate([discrete_action,full_action[self.acs_discrete:]])


            return full_action.tolist()
        else:
            return full_action
    '''def find_best_action(self, network, observation: np.ndarray) -> np.ndarray:
        if self.action_policy =='argmax':
            return np.random.choice(network(torch.tensor(observation).float()).detach().numpy())
        else:
            return misc.action2one_hot(self.acs_space,np.argmax(network(torch.tensor(observation).float()).detach()).item())'''

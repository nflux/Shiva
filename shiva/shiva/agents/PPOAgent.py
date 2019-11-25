from .Agent import Agent
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
import utils.Noise as noise
import copy
import torch
import numpy as np
import helpers.misc as misc

class PPOAgent(Agent):
    def __init__(self, id, obs_dim, acs_discrete, acs_continuous, agent_config,net_config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.acs_discrete = acs_discrete
        self.acs_continuous = acs_continuous

        if (acs_discrete != None) and (acs_continuous != None):
            self.network_output = self.acs_discrete + self.acs_continuous
        elif (self.acs_discrete == None):
            self.network_output = self.acs_continuous
        else:
            self.network_output = self.acs_discrete
        self.id = id
        self.network_input = obs_dim

        self.actor = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config['actor'])
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
        #self.target_critic = copy.deepcopy(self.critic)

        params = list(self.actor.parameters()) +list(self.critic.parameters())
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=params, lr=agent_config['learning_rate'])


    def get_action(self,observation):
        #retrieve the action given an observation
        full_action = self.target_actor(torch.tensor(observation).float()).detach().numpy()
        if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))

        #Check if there is a discrete component to the action tensor
        if (self.acs_discrete != None):
            #If there is, extract the discrete component
            discrete_action = full_action[: self.acs_discrete]
            dist = Categorical(torch.tensor(discrete_action).float())
            discrete_action = dist.sample()
            discrete_action = misc.action2one_hot(self.acs_discrete,discrete_action.item())

            #Recombine the chosen discrete action with the continuous action values
            full_action = np.concatenate([discrete_action,full_action[self.acs_discrete:]])

            return full_action.tolist()
        else:
            return full_action


    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic, save_path +'/critic.pth')

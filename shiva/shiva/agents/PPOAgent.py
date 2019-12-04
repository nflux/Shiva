from .Agent import Agent
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
from torch.nn import functional as F
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
        self.scale = 0.9


        if (acs_discrete != None) and (acs_continuous != None):
            self.network_output = self.acs_discrete + self.acs_continuous
            self.ou_noise = noise.OUNoise(acs_discrete+acs_continuous, self.scale)
        elif (self.acs_discrete == None):
            self.network_output = self.acs_continuous
            self.ou_noise = noise.OUNoise(acs_continuous, self.scale)
        else:
            self.network_output = self.acs_discrete
            self.ou_noise = noise.OUNoise(acs_discrete, self.scale)
        self.id = id
        self.network_input = obs_dim

        if agent_config['action_space'] == 'Discrete':
            self.actor = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config['actor'])
            #self.target_actor = copy.deepcopy(self.actor)
            self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
            params = list(self.actor.parameters()) +list(self.critic.parameters())
            self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=params, lr=agent_config['learning_rate'])

        elif agent_config['action_space'] == 'Continuous':
                self.policy_base = torch.nn.Sequential(
                    torch.nn.Linear(self.network_input,net_config['policy_base_output']),
                    torch.nn.ReLU())
                self.mu = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['mu'])
                self.actor = self.mu
                self.var = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['var'])
                self.critic = DLN.DynamicLinearNetwork(net_config['policy_base_output'],1,net_config['critic'])
                params = list(self.policy_base.parameters()) + list(self.mu.parameters()) + list(self.var.parameters()) + list(self.critic.parameters())
                self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=params, lr=agent_config['learning_rate'])


    def get_action(self,observation):
        #retrieve the action given an observation
        action = self.actor(torch.tensor(observation).float()).detach()
        '''if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))'''
        dist = Categorical(action)
        action = dist.sample()
        action = misc.action2one_hot(self.acs_discrete,action.item())
        return action.tolist()

    def get_continuous_action(self,observation):
        observation = torch.tensor(observation).float().detach()
        base_output = self.policy_base(observation)
        mu = self.mu(base_output).detach().numpy()
        sigma = torch.sqrt(self.var(base_output)).detach().numpy()
        actions = np.random.normal(mu,sigma)
        self.ou_noise.set_scale(0.1)
        actions += self.ou_noise.noise()
        actions = np.clip(actions,-1,1)
        return actions.tolist()

    def evaluate(self,observation):
        observation = torch.tensor(observation).float().detach()
        base_output = self.policy_base(observation)
        mu = self.mu(base_output)
        var = self.var(base_output)
        value = self.critic(base_output)
        return my, var, value


    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic, save_path +'/critic.pth')

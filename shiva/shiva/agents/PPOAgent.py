from .Agent import Agent
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.nn import functional as F
import torch.nn as nn
import utils.Noise as noise
import copy
import torch
import numpy as np
import helpers.misc as misc

def init_layer(layer):
    w = layer.weight.data
    w.normal_(0,1)
    w *= 1.0 / (torch.sqrt(w.pow(2).sum(1,keepdim=True)))
    torch.nn.init.constant_(layer.bias.data,0)
    return layer

class PPOAgent(Agent):
    def __init__(self, id, obs_dim, acs_discrete, acs_continuous, agent_config, net_config):
        super(PPOAgent, self).__init__(id, obs_dim, (0 if acs_discrete is None else acs_discrete)+(0 if acs_continuous is None else acs_continuous), agent_config, net_config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(agent_config['manual_seed'])
        np.random.seed(agent_config['manual_seed'])
        self.acs_discrete = acs_discrete
        self.acs_continuous = acs_continuous
        self.scale = 0.9
        print('Action_Space: ', self.action_space)


        if (acs_discrete != None) and (acs_continuous != None):
            # parametrized PPO?
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

        if self.action_space == 'Discrete':
            print('actor:', self.network_input, self.network_output)
            self.actor = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config['actor'])
            #self.target_actor = copy.deepcopy(self.actor)
            self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
            self.params = list(self.actor.parameters()) +list(self.critic.parameters())
            self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.params, lr=agent_config['learning_rate'])

        elif self.action_space == 'Continuous':
                '''self.policy_base = torch.nn.Sequential(
                    torch.nn.Linear(self.network_input, net_config['policy_base_output']),
                    torch.nn.ReLU()
                ).to(self.device)
                self.mu = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['mu'])
                self.actor = self.mu
                self.var = DLN.DynamicLinearNetwork(net_config['policy_base_output'], self.network_output, net_config['var'])
                self.critic = DLN.DynamicLinearNetwork(net_config['policy_base_output'], 1, net_config['critic'])'''
                self.actor = nn.Sequential(
                    init_layer(nn.Linear(self.network_input,128)),
                    nn.Tanh(),
                    init_layer(nn.Linear(128,128)),
                    nn.Tanh(),
                    init_layer(nn.Linear(128,self.network_output))
                )
                self.critic = nn.Sequential(
                    init_layer(nn.Linear(self.network_input,128)),
                    nn.Tanh(),
                    init_layer(nn.Linear(128,128)),
                    nn.Tanh(),
                    init_layer(nn.Linear(128,1)),
                )
                #self.mu = init_layer(nn.Linear(64,self.network_output))
                #self.log_std = torch.zeros(1,self.network_output).requires_grad_(True)
                self.log_std = nn.Parameter(torch.zeros(1,self.network_output))
                self.params = list(self.actor.parameters()) + list(self.critic.parameters())
                self.optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.parameters(), lr=agent_config['learning_rate'],eps=agent_config['eps'])


    def forward(self, observation):
        if self.action_space == 'Discrete':
            return self.get_discrete_action(observation)
        elif self.action_space == 'Continuous':
            return self.get_continuous_action(observation)

    def get_discrete_action(self, observation):
        #retrieve the action given an observation
        action = self.actor(torch.tensor(observation).float()).detach()
        '''if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))'''
        dist = Categorical(action)
        action = dist.sample()
        action = misc.action2one_hot(self.acs_discrete, action.item())
        return action.tolist()

    def get_continuous_action(self,observation):
        observation = torch.tensor(observation).float().detach().to(self.device)
        mu = self.actor(observation)
        log_std = self.log_std
        dist = Normal(mu,log_std)
        actions = dist.sample()
        #self.ou_noise.set_scale(0.8)
        #actions += self.ou_noise.noise()
        #actions = np.clip(actions,-1,1)
        return actions.squeeze(dim=0).tolist()

    def evaluate(self,observation):
        observation = torch.tensor(observation).float().detach()
        base_output = self.policy_base(observation)
        mu = self.mu(base_output)
        var = self.var(base_output)
        value = self.critic(base_output)
        return mu, var, value


    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic, save_path +'/critic.pth')

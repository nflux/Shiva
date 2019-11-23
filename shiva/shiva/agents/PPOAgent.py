from .Agent import Agent
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
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
        self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
        self.affine = nn.Linear(self.network_input,net_config['affine'])


        #self.optimizer_actor = getattr(torch.optim,agent_config['optimizer_function'])(params=self.actor.parameters(), lr=agent_config['learning_rate'])
        self.optimizer_critic = getattr(torch.optim,agent_config['optimizer_function'])(params=self.critic.parameters(), lr=agent_config['learning_rate'])
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.parameters(), lr=agent_config['learning_rate'])


    def get_action(self,observation):
        observation = torch.tensor(observation).clone().detach().float().requires_grad_(True)
        action_probs = self.actor(observation)
        #action_probs = self.actor(torch.from_numpy(observation).float()).detach().requires_grad_(True)
        dist = Categorical(action_probs)
        action = dist.sample()
        if action.dim() > 0:
            action_one_hot = [None] * len(observation)
            for i in range(len(action_one_hot)):
                action_one_hot[i] = misc.action2one_hot(self.acs_discrete,action[i].item()).tolist()
            return action_one_hot
        else:
            action = misc.action2one_hot(self.acs_discrete,action.item()).tolist()
            return action



    def get_logprobs(self,observations,actions):
        logprobs = [None] * len(actions)
        entropy = [None] * len(actions)
        acs_discrete = [None] * len(actions)
        for i in range(len(actions)):
            acs_discrete[i]= np.argmax(actions[i])
        action_probs = self.actor(torch.tensor(observations).float()).detach()
        for i in range(len(logprobs)):
            dist = Categorical(action_probs[i])
            logprobs[i] = dist.log_prob(torch.tensor(acs_discrete[i]).clone().detach())
            entropy[i] = dist.log_prob(torch.tensor(acs_discrete[i]).clone().detach())
            #logprobs[i] = logprob
        return logprobs,entropy

    def get_logprob(self,observation, action):
        action = np.argmax(action)
        action_probs = self.actor(torch.tensor(observation).float()).detach()
        dist = Categorical(action_probs)
        logprob = dist.log_prob(torch.tensor(action).clone().detach())
        return logprob

    def get_values(self,observation,action):
        values = self.critic(torch.tensor(observation).float()).detach()
        return values

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic, save_path +'/critic.pth')

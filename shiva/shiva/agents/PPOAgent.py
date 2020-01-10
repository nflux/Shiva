import copy
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from torch.nn import Softmax as Softmax
import utils.Noise as noise
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from torch.nn import functional as F

from shiva.agents.Agent import Agent
from shiva.networks import DynamicLinearNetwork as DLN
from shiva.utils import Noise as noise
from shiva.helpers import misc

def init_layer(layer):
    weights = layer.weight.data
    weights.normal_(0,1)
    weights *= 1.0 / torch.sqrt(weights.pow(2).sum(1,keepdim=True))
    nn.init.constant_(layer.bias.data,0)
    return layer

class PPOAgent(Agent):
    def __init__(self, id, obs_dim, acs_space, agent_config, net_config):
        super(PPOAgent, self).__init__(id, obs_dim, acs_space['acs_space'], agent_config, net_config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(agent_config['manual_seed'])
        np.random.seed(agent_config['manual_seed'])

        self.acs_discrete = acs_space['discrete']
        self.acs_continuous = acs_space['param']
        self.scale = 0.9
        self.softmax = Softmax(dim=-1)

        if (self.acs_discrete != 0) and (self.acs_continuous != 0):
            # parametrized PPO?
            self.action_space = 'Param'
            self.network_output = self.acs_discrete + self.acs_continuous
            self.ou_noise = noise.OUNoise(self.acs_discrete+self.acs_continuous, self.scale)
        elif (self.acs_discrete == 0):
            self.action_space = 'Continuous'
            self.network_output = self.acs_continuous
            self.ou_noise = noise.OUNoise(self.acs_continuous, self.scale)
        else:
            self.action_space = 'Discrete'
            self.network_output = self.acs_discrete
            self.ou_noise = noise.OUNoise(self.acs_discrete, self.scale)
            
        self.id = id
        self.network_input = obs_dim

        if self.action_space == 'Discrete':
            print('actor:', self.network_input, self.network_output)
            self.actor = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config['actor'])
            #self.target_actor = copy.deepcopy(self.actor)
            self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
            params = list(self.actor.parameters()) +list(self.critic.parameters())
            self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=params, lr=agent_config['learning_rate'])

        elif self.action_space == 'Continuous':
            '''self.policy_base = torch.nn.Sequential(
                init_layer(torch.nn.Linear(self.network_input, net_config['policy_base_output'])),
                torch.nn.ReLU())
            self.mu = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['mu'])
            self.actor = self.mu
            self.log_std = nn.Parameter(torch.zeros(1,self.network_output))
            #self.var = torch.full((self.network_output,),self.std**2,requires_grad=True).to(self.device)
            self.critic = DLN.DynamicLinearNetwork(net_config['policy_base_output'], 1, net_config['critic'])
            self.actor_params = list(self.mu.parameters())  + list(self.critic.parameters())+ list(self.log_std.parameters())
            self.optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.params, lr=agent_config['learning_rate'])'''
            self.policy_base = torch.nn.Sequential(
                init_layer(torch.nn.Linear(self.network_input, net_config['policy_base_output'])),
                torch.nn.ReLU())
            self.mu = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['mu'])
            self.actor = self.mu
            self.var = DLN.DynamicLinearNetwork(net_config['policy_base_output'],self.network_output,net_config['var'])
            self.critic = DLN.DynamicLinearNetwork(net_config['policy_base_output'],1,net_config['critic'])
            self.critic_params = list(self.critic.parameters())
            self.actor_params = list(self.mu.parameters()) + list(self.var.parameters()) + list(self.policy_base.parameters())
            self.actor_optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.actor_params, lr=agent_config['learning_rate'])
            self.critic_optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.critic_params, lr=agent_config['learning_rate'])


    def get_action(self, observation):
        if self.action_space == 'Discrete':
            return self.get_discrete_action(observation)
        elif self.action_space == 'Continuous':
            return self.get_continuous_action(observation)

    def get_logprobs(self,observation,action):
        if self.action_space == 'Discrete':
            return self.get_discrete_logprobs(observation,action)
        elif self.action_space == 'Continuous':
            return self.get_continuous_logprobs(observation,action)


    def evaluate(self,observation):
        if self.action_space == 'Discrete':
            return self.evaluate_discrete(observation)
        elif self.action_space == 'Continuous':
            return self.get_continuous_action(observation)

    def get_discrete_action(self, observation):
        #retrieve the action given an observation
        action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
        action = self.softmax(action)
        '''if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))'''
        dist = Categorical(action)
        action = dist.sample()
        action = misc.action2one_hot(self.acs_discrete, action.item())
        return action.tolist()


    def evaluate_discrete(self,observation):
        action = self.actor(torch.tensor(observation).float()).detach()
        dist = Categorical(action)
        action = dist.sample()
        logprobs = dist.log_prob(action)
        action = misc.action2one_hot(self.acs_discrete, action.item())
        return action.tolist(), logprobs.tolist()

    def get_discrete_logprobs(self,observation,action):
        action_probs = self.actor(torch.tensor(observation).float()).detach()
        dist = Categorical(action_probs)
        action = torch.tensor(np.argmax(action, axis=-1)).to(self.device).long()
        logprobs = dist.log_prob(action)
        return logprobs.tolist()

    '''def get_continuous_action(self,observation):
        observation = torch.tensor(observation).float().detach().to(self.device)
        base_output = self.policy_base(observation)
        mu = self.mu(base_output).detach().cpu()
        cov_mat = torch.diag(self.var)
        dist = MultivariateNormal(mu,cov_mat)
        action = dist.sample()
        self.ou_noise.set_scale(0.8)
        action += torch.tensor(self.ou_noise.noise()).float()
        action = torch.clamp(action,-1,1)
        return action.tolist()'''

    '''def get_continuous_action(self,observation):
        observation = torch.tensor(observation).float().detach().to(self.device)
        base_output = self.policy_base(observation)
        mu = self.mu(base_output).to(self.device)
        sigma = torch.sqrt(self.var(base_output)).to(self.device)
        #actions = np.random.normal(mu,sigma)
        actions = Normal(mu,sigma).sample()
        self.ou_noise.set_scale(0.8)
        actions +=torch.tensor(self.ou_noise.noise()).float()
        actions = np.clip(actions,-1,1)
        return actions.tolist()'''

    def get_continuous_action(self,observation):
        observation = torch.tensor(observation).float().detach().to(self.device)
        base = self.policy_base(observation).to(self.device)
        mu = self.mu(base).to(self.device)
        sigma = torch.sqrt(self.var(base)).to(self.device)
        #actions = np.random.normal(mu,sigma)
        actions = Normal(mu,sigma).sample()
        #self.ou_noise.set_scale(0.8)
        #actions +=torch.tensor(self.ou_noise.noise()).float()
        actions = np.clip(actions,-1,1)
        return actions.tolist()

    def get_continuous_logprobs(self,observation,action):
        observation = torch.tensor(observation).float().detach().to(self.device)
        base = self.policy_base(observation).to(self.device)
        action = torch.tensor(action).float().to(self.device)
        mu = self.mu(base).to(self.device)
        sigma = torch.sqrt(self.var(base)).to(self.device)
        dist = Normal(mu,sigma)
        logprobs = dist.log_prob(action)
        return logprobs.tolist()

    '''def evaluate(self,observation):
        observation = torch.tensor(observation).float().detach()
        base_output = self.policy_base(observation)
        mu = self.mu(base_output)
        var = self.var(base_output)
        value = self.critic(base_output)
        return my, var, value'''


    def save_agent(self, save_path,step):

        '''torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'agent' : self
        }, save_path + '/agent.pth')'''
        torch.save(self.actor.state_dict(), save_path + '/actor.pth')
        torch.save(self.critic.state_dict(), save_path +'/critic.pth')
        torch.save(self,save_path + '/agent.pth')

    def save(self,save_path,step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic,save_path + '/critic.pth')

    def load(self,save_path):
        # print(save_path)
        '''model = torch.load(save_path + '/agent.pth')
        self.actor.load_state_dict( model['actor'])
        self.critic.load_state_dict(model['critic'])'''
        self = torch.load(save_path+'/agent.pth')
        self.actor.load_state_dict(torch.load(save_path+'/actor.pth'))
        self.critic.load_state_dict(torch.load(save_path+'/critic.pth'))

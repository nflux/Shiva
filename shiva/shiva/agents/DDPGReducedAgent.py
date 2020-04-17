import numpy as np
import random
import torch
import copy
import pickle
from torch.distributions import Categorical
from torch.nn.functional import softmax
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.Agent import Agent
from shiva.utils import Noise as noise
from shiva.helpers.misc import action2one_hot
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork

class DDPGReducedAgent(Agent):
    def __init__(self, id:int, obs_space:int, acs_space:dict, agent_config: dict, networks: dict):
        super(DDPGReducedAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)
        self.agent_config = agent_config
        #try:
            #torch.manual_seed(self.manual_seed)
            #np.random.seed(self.manual_seed)
        #except:
            #torch.manual_seed(5)
            #np.random.seed(5)
            #torch.manual_seed(self.id)
            #np.random.seed(self.id)

        self.discrete = acs_space['discrete']
        self.continuous = acs_space['continuous']
        self.param = acs_space['discrete']
        #self.acs_space = acs_space['acs_space']

        #if self.continuous == 0:
            #self.action_space = 'discrete'
            # print(self.action_space)
            #self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.discrete, self.param, networks['actor'])
        #elif self.discrete == 0:
            #self.action_space = 'continuous'
            # print(self.action_space)
            #self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.continuous, self.param, networks['actor'])
        #else:
            #print("DDPG Agent, check if this makes sense for parameterized robocup")
            #self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.discrete+self.param, self.param, networks['actor'])

        #self.target_actor = copy.deepcopy(self.actor)

        #self.critic = DynamicLinearNetwork(obs_space + self.acs_space, 1, networks['critic'])
        #self.target_critic = copy.deepcopy(self.critic)

        if self.continuous == 0:
            self.action_space = 'discrete'
            self.actor_input = obs_space
            self.actor_output = self.discrete
            self.get_action = self.get_discrete_action
        elif self.discrete == 0:
            self.action_space = 'continuous'
            self.actor_input = obs_space
            self.actor_output = self.continuous
            self.get_action = self.get_continuous_action
        else:
            self.action_space = 'parametrized'
            self.actor_input = obs_space
            self.actor_output = self.discrete + self.param
            self.get_action = self.get_parameterized_action

        if hasattr(self, 'lr_range') and self.lr_range:
            self.critic_learning_rate = np.random.uniform(agent_config['lr_uniform'][0],agent_config['lr_uniform'][1]) / np.random.choice(agent_config['lr_factors'])
            self.actor_learning_rate = np.random.uniform(agent_config['lr_uniform'][0],agent_config['lr_uniform'][1]) / np.random.choice(agent_config['lr_factors'])
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
            self.ou_noise = noise.OUNoise(self.actor_output, self.noise_scale)
        else:
            self.actor_learning_rate = agent_config['actor_learning_rate']
            self.critic_learning_rate = agent_config['critic_learning_rate']
            self.ou_noise = noise.OUNoise(self.acs_space, self.exploration_noise)

        if hasattr(self,'rewards') and self.rewards: # Flag saying whether use are optimizing reward functions with PBT
            self.set_reward_factors()

        self.instantiate_networks()

    def instantiate_networks(self):
        self.net_names = ['actor']
        self.actor = SoftMaxHeadDynamicLinearNetwork(self.actor_input, self.actor_output, self.param, self.networks_config['actor']).to(self.device)
        torch.manual_seed(self.manual_seed)
        np.random.seed(self.manual_seed)

    def to_device(self, device):
        self.device = device
        self.actor.to(self.device)


    def get_action(self, observation, step_count, evaluate=False):
        if self.action_space == 'discrete':
            return self.get_discrete_action(observation, step_count, evaluate)
        elif self.action_space == 'continuous':
            return self.get_continuous_action(observation, step_count, evaluate)
        elif self.action_space == 'parameterized':
            assert "DDPG Parametrized NotImplemented"
            pass
            return self.get_parameterized_action(observation, evaluate)

    def get_discrete_action(self, observation, step_count, evaluate):
        #if evaluate:
            #action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            #self.ou_noise.set_scale(self.noise_scale)
            #action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
            #action = torch.abs(action)
            #action = action / action.sum()
            # print("Agent Evaluate {}".format(action))
        #else:
        if hasattr(self, 'epsilon') and (np.random.uniform(0, 1) < self.epsilon):
            action = np.array([np.random.uniform(0, 1) for _ in range(self.actor_output)])
            action = softmax(torch.from_numpy(action), dim=-1)
        else:
            if hasattr(self,'noise_scale'):
                self.ou_noise.set_scale(self.noise_scale)
            else:
                self.ou_noise.set_scale(self.training_noise)
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
            action = torch.clamp(action, 0,1)
            action = action / action.sum()
            # print("Net: {}".format(action))
        return action.tolist()

    def get_continuous_action(self,observation, step_count, evaluate):
        if evaluate:
            observation = torch.tensor(observation).float().to(self.device)
            action = self.actor(observation)
        else:
            if step_count < self.exploration_steps:
                self.ou_noise.set_scale(self.exploration_noise)
                action = np.array([np.random.uniform(0,1) for _ in range(self.actor_output)])
                action += self.ou_noise.noise()
                action = softmax(torch.from_numpy(action))

            else:
                observation = torch.tensor(observation).float().to(self.device)
                action = self.actor(observation)
                self.ou_noise.set_scale(self.training_noise)
                action += torch.tensor(self.ou_noise.noise()).float().to(self.device)
                action = action/action.sum()

        return action.tolist()

    def get_parameterized_action(self, observation, step_count, evaluate):
        if evaluate:
            observation = torch.tensor(observation).to(self.device)
            action = self.actor(observation.float())
        else:
            # if step_count > self.exploration_steps:
            # else:
            observation = torch.tensor(observation).to(self.device)
            action = self.actor(observation.float())
        # return action.tolist()

        return action[0]


    def get_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float())
        return action[0]



    def set_reward_factors(self):
        self.reward_factors = dict()
        for reward in self.reward_events:
            self.reward_factors[reward] = np.random.uniform(self.reward_range[0],self.reward_range[1])


    def get_module_and_classname(self):
        return ('shiva.agents', 'DDPGReducedAgent.DDPGReducedAgent')

    def __str__(self):
        if self.rewards:
            return '<DDPGAgent(id={}, steps={}, episodes={}, reward_factors={})>'.format(self.id, self.step_count, self.done_count,self.reward_factors)
        else:
            return '<DDPGAgent(id={}, step{}, episodes={})>'.format(self.id,self.step_count, self.done_count)

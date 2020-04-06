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

class DDPGAgent(Agent):
    def __init__(self, id:int, obs_space:int, acs_space:dict, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)
        self.agent_config = agent_config
        try:
            torch.manual_seed(self.manual_seed)
            np.random.seed(self.manual_seed)
        except:
            #torch.manual_seed(5)
            #np.random.seed(5)
            self.seed = np.random.randint(0, 100)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)


        self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
        if self.rewards: # Flag saying whether use are optimizing reward functions with PBT
            self.set_reward_factors()

        self.id = id

        '''

            Maybe do something like


        '''
        self.discrete = acs_space['discrete']
        self.continuous = acs_space['continuous']
        self.param = acs_space['discrete']
        self.acs_space = acs_space['acs_space']
        self.step_count = 0
        self.done_count = 0

        if self.continuous == 0:
            self.action_space = 'discrete'
            # print(self.action_space)
            self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.discrete, self.param, networks['actor'])
        elif self.discrete == 0:
            self.action_space = 'continuous'
            # print(self.action_space)
            self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.continuous, self.param, networks['actor'])
        else:
            print("DDPG Agent, check if this makes sense for parameterized robocup")
            self.actor = SoftMaxHeadDynamicLinearNetwork(obs_space,self.discrete+self.param, self.param, networks['actor'])

        self.target_actor = copy.deepcopy(self.actor)

        self.critic = DynamicLinearNetwork(obs_space + self.acs_space, 1, networks['critic'])
        self.target_critic = copy.deepcopy(self.critic)


        if agent_config['lr_range']:
            self.critic_learning_rate = np.random.uniform(agent_config['lr_uniform'][0],agent_config['lr_uniform'][1]) / np.random.choice(agent_config['lr_factors'])
            self.actor_learning_rate = np.random.uniform(agent_config['lr_uniform'][0],agent_config['lr_uniform'][1]) / np.random.choice(agent_config['lr_factors'])
            self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
            self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)
        else:
            self.actor_learning_rate = agent_config['actor_learning_rate']
            self.critic_learning_rate = agent_config['critic_learning_rate']
            self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
            self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

        self.ou_noise = noise.OUNoise(self.acs_space, self.exploration_noise)



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
        if step_count < self.exploration_steps:
            self.ou_noise.set_scale(self.noise_scale)
            action = np.array([np.random.uniform(0,1) for _ in range(self.acs_space)])
            action = torch.from_numpy(action + self.ou_noise.noise())
            action = softmax(action, dim=-1)
            # print("Random: {}".format(action))
        elif np.random.uniform(0, 1) < self.epsilon:
            action = np.array([np.random.uniform(0, 1) for _ in range(self.acs_space)])
            action = softmax(torch.from_numpy(action), dim=-1)
        else:
            self.ou_noise.set_scale(self.noise_scale)
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
            action = torch.abs(action)
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
                action = np.array([np.random.uniform(0,1) for _ in range(self.acs_space)])
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

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')
        torch.save(self.actor_optimizer, save_path + '/actor_optimizer.pth')
        torch.save(self.critic_optimizer, save_path + '/critic_optimizer.pth')

    # def load(self,save_path):
    #     self.actor.load_state_dict(torch.load(save_path + 'actor.pth'))
    #     self.target_actor.load_state_dict(torch.load(save_path + 'target_actor.pth'))
    #     self.critic.load_state_dict(torch.load(save_path + 'critic.pth'))
    #     self.target_critic.load_state_dict(torch.load(save_path + 'target_critic.pth'))
    #     self.actor_optimizer.load_state_dict(torch.load(save_path + 'actor_optimizer.pth'))
    #     self.critic_optimizer.load_state_dict(torch.load(save_path + 'critic_optimizer.pth'))

    def set_reward_factors(self):
        self.reward_factors = dict()
        for reward in self.reward_events:
            self.reward_factors[reward] = np.random.uniform(self.reward_range[0],self.reward_range[1])

    def copy_weights(self,evo_agent):
        self.actor_learning_rate = evo_agent.actor_learning_rate
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self.actor_optimizer = copy.deepcopy(evo_agent.actor_optimizer)
        self.critic_optimizer = copy.deepcopy(evo_agent.critic_optimizer)
        self.actor = copy.deepcopy(evo_agent.actor)
        self.target_actor = copy.deepcopy(evo_agent.target_actor)
        self.critic = copy.deepcopy(evo_agent.critic)
        self.target_critic = copy.deepcopy(evo_agent.target_critic)
        self.epsilon = evo_agent.epsilon
        self.noise_scale = evo_agent.noise_scale
        self.reward_factors = evo_agent.reward_factors

    def perturb_hyperparameters(self,perturb_factor):
        perturb_factor = np.random.choice(self.perturb_factors)
        self.actor_learning_rate = self.actor_learning_rate * perturb_factor
        self.critic_learning_rate = self.critic_learning_rate * perturb_factor
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_learning_rate

        self.epsilon *= perturb_factor
        self.noise_scale *= perturb_factor
        if self.rewards:
            for reward in self.reward_factors:
                self.reward_factors[reward] *= perturb_factor
        #self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        #self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

    def resample_hyperparameters(self):
        self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0],self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0],self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_learning_rate
        self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
        if self.rewards:
            for reward in self.reward_factors:
                self.reward_factors[reward] = np.random.uniform(self.reward_range[0],self.reward_range[1])
        #self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        #self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

    def robocup_exploration(self):
        perturb_prob = 1 / self.num_adaptable_params
        resample_prob = 1 / self.num_adaptable_params
        keep_prob = 1 - perturb_prob - resample_prob
        options = ['pass','resample','perturb']
        option_probs = [keep_prob,resample_prob,perturb_prob]
        print('Option Probs: {}'.format(option_probs))

        choice = np.random.choice(options,size = self.num_adaptable_params,p=option_probs)
        if choice[0] == 'resample':
            self.resample_actor_learning_rate()
        elif choice[0] == 'perturb':
            self.perturb_actor_learning_rate()

        if choice[1] =='resample':
            self.resample_critic_learning_rate()
        elif choice[1] == 'perturb':
            self.perturb_critic_learning_rate()

        if choice[2] == 'resample':
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        elif choice[2] == 'perturb':
            perturb_factor = np.random.choice(self.perturb_factors)
            self.epsilon *= perturb_factor

        if choice[3] == 'resample':
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
        elif choice[3] == 'perturb':
            perturb_factor = np.random.choice(self.perturb_factors)
            self.noise_scale *= perturb_factor

        self.reward_exploration(choice[4:])

    def resample_actor_learning_rate(self):
        self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0],self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_learning_rate

    def perturb_actor_learning_rate(self):
        perturb_factor = np.random.choice(self.perturb_factors)
        self.actor_learning_rate = self.actor_learning_rate * perturb_factor
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_learning_rate


    def resample_critic_learning_rate(self):
        self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0],self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_learning_rate

    def perturb_critic_learning_rate(self):
        perturb_factor = np.random.choice(self.perturb_factors)
        self.critic_learning_rate = self.critic_learning_rate * perturb_factor
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_learning_rate


    def reward_exploration(self,choices):
        for i,reward in enumerate(self.reward_factors.keys()):
            if choices[i] == 'resample':
                self.reward_factors[reward] = np.random.uniform(self.reward_range[0],self.reward_range[1])
            elif choices[i] == 'perturb':
                perturb_factor = np.random.choice(self.perturb_factors)
                self.reward_factors[reward] *= perturb_factor




    def __str__(self):
        return '<DDPGAgent(id={}, steps={}, episodes={})>'.format(self.id, self.step_count, self.done_count)

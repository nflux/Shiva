import copy, os
import numpy as np
import torch

from shiva.agents.Agent import Agent
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork

class TD3Agent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(TD3Agent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        try:
            torch.manual_seed(self.manual_seed)
            np.random.seed(self.manual_seed)
        except:
            torch.manual_seed(5)
            np.random.seed(5)

        self.id = id

        self.actor = DynamicLinearNetwork(obs_dim, action_dim, networks['actor'])
        self.target_actor = DynamicLinearNetwork(obs_dim, action_dim, networks['actor'])
        self.copy_model_over(self.actor, self.target_actor)

        self.critic = DynamicLinearNetwork(obs_dim + action_dim, 1, networks['critic'])
        self.target_critic = DynamicLinearNetwork(obs_dim + action_dim, 1, networks['critic'])
        self.copy_model_over(self.critic, self.target_critic)

        self.critic_2 = DynamicLinearNetwork(obs_dim + action_dim, 1, networks['critic_2'])
        self.target_critic_2 = DynamicLinearNetwork(obs_dim + action_dim, 1, networks['critic_2'])
        self.copy_model_over(self.critic_2, self.target_critic_2)
        
        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_lr, eps=self.eps)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_lr, eps=self.eps)
        self.critic_optimizer_2 = self.optimizer_function(params=self.critic_2.parameters(), lr=self.critic_2_lr, eps=self.eps)

    def get_action(self, observation):
        return self.actor(observation)

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')
        torch.save(self.critic_2, save_path + '/critic_2.pth')
        torch.save(self.target_critic_2, save_path + '/target_critic_2.pth')

    def load_net(self, load_path):
        network = torch.load(load_path)
        attr = os.path.split('/')[-1].replace('.pth', '')
        setattr(self, attr, network)
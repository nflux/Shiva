import copy, os
import numpy as np
import torch

from shiva.agents.Agent import Agent
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork
from shiva.utils import Noise as noise

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

        self.ou_noise = noise.OUNoise(self.acs_space, self.noise_scale, self.noise_mu, self.noise_theta, self.noise_sigma)
        self.ou_noise_critic = noise.OUNoise(self.acs_space, self.noise_scale, self.noise_mu, self.noise_theta, self.noise_sigma)

        self.action = self.get_random_action()

    def get_action(self, observation, step_count):
        if not torch.is_tensor(observation):
            observation = torch.tensor(observation).float()

        if step_count < self.exploration_steps:
            '''Exploration random action'''
            # check if obs is a batch!
            if len(observation.shape) > 1:
                # print('random batch action')
                action = [self.get_random_action() for _ in range(observation.shape[0])]
                self.action = action # for tensorboard
            else:
                # print("random act")
                action = self.get_random_action()
                self.action = [action] # for tensorboard
        else:
            """Picks an action using the actor network and then adds some noise to it to ensure exploration"""
            self.actor.eval()
            with torch.no_grad():
                obs = torch.tensor(observation, dtype=torch.float).to(self.device)
                action = self.actor(obs).cpu().data.numpy()
            self.actor.train()
            if len(observation.shape) > 1:
                # print('batch action')
                action = [list(act + self.ou_noise.noise()) for act in action]
                self.action = action # for tensorboard
            else:
                # print('single action')
                action += self.ou_noise.noise()
                action = action.tolist()
                self.action = [action] # for tensorboard
        return action

    def get_random_action(self):
        return np.array([np.random.uniform(-1, 1) for _ in range(self.acs_space)]).tolist()

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')
        torch.save(self.critic_2, save_path + '/critic_2.pth')
        torch.save(self.target_critic_2, save_path + '/target_critic_2.pth')

    # def load_net(self, load_path):
    #     network = torch.load(load_path)
    #     attr = os.path.split('/')[-1].replace('.pth', '')
    #     setattr(self, attr, network)

    def __str__(self):
        return 'TD3Agent'
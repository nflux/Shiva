import numpy as np
np.random.seed(5)
import torch
torch.manual_seed(5)
from .Agent import Agent
import copy
from networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork

class TD3Agent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(TD3Agent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
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
        
    def find_best_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float()).cpu().data.numpy()
        action = np.clip(action, -1,1)
        # print('actor action shape', action.shape)
        return action[0]
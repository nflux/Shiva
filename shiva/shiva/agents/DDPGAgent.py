from .Agent import Agent
import networks.DDPGActor as actor
import networks.DDPGCritic as critic
import copy
import torch
import numpy as np
import helpers.misc as misc

from networks.DynamicLinearNetwork import SoftMaxHeadDynamicLinearNetwork


class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        self.id = id

        # print("Look here: ", networks)

        # print("DDPG Agent:", obs_dim, action_dim)


        self.actor = SoftMaxHeadDynamicLinearNetwork(obs_dim,
                                    action_dim,
                                    action_dim,
                                    networks['actor'])

        self.target_actor = copy.deepcopy(self.actor)

        self.critic = critic.DDPGCritic(obs_dim,
                                        action_dim,
                                        networks['critic_head'],
                                        networks['critic_tail'])

        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.learning_rate)


        # Uncomment to check the networks

        # print(self.actor)

        # print(self.critic)

        # input()


    def find_best_imitation_action(self, observation: np.ndarray) -> np.ndarray:

        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float()).cpu().data.numpy()
        action = np.clip(action, -1,1)

        return action

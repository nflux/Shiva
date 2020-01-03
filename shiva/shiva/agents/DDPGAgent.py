import copy
import torch
import numpy as np

from shiva.agents.Agent import Agent
from shiva.networks.DDPGActor import DDPGActor
from shiva.networks.DDPGCritic import DDPGCritic
from shiva.helpers import misc

class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        self.id = id

        # print("Look here: ", networks)

        # print("DDPG Agent:", obs_dim, action_dim)


        self.actor = DDPGActor(obs_dim,
                                    action_dim,
                                    networks['actor'])

        self.target_actor = copy.deepcopy(self.actor)

        self.critic = DDPGCritic(obs_dim,
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

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')
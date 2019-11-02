from .Agent import Agent
import copy
from networks.DynamicLinearNetwork import DynamicLinearNetwork

class ParametrizedDDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(ParametrizedDDPGAgent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        self.id = id

        self.actor = DynamicLinearNetwork(obs_dim, action_dim, networks)
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = DynamicLinearNetwork(obs_dim + action_dim, 1, networks)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.learning_rate)


        # Uncomment to check the networks

        # print(self.actor)

        # print(self.critic)

        # input()
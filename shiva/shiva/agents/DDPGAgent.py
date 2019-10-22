from .Agent import Agent
import networks.DDPGActor as actor
import networks.DDPGCritic as critic
import copy

class DDPGAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DDPGAgent, self).__init__(obs_dim, action_dim, optimizer, learning_rate, config)

        self.actor = actor.DDPGActor(obs_dim, action_dim, config['network']['network_actor'])
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = critic.DDPGCritic(obs_dim, action_dim, config['network']['network_critic_head'], config['network']['network_critic_tail'])
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=learning_rate)
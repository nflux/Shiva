from .Agent import Agent
import networks.DDPGActor as actor
import networks.DDPGCritic as critic
import copy

class DDPGAgent(Agent):
    def __init__(self, agent_id, obs_dim, action_dim, agent_config: dict, network_config: dict):
        super(DDPGAgent, self).__init__(agent_id, obs_dim, action_dim, agent_config, network_config)

        self.actor = actor.DDPGActor(obs_dim, 
                                    action_dim, 
                                    self.network_config['network']['network_actor'])

        self.target_actor = copy.deepcopy(self.actor)

        self.critic = critic.DDPGCritic(obs_dim, 
                                        action_dim, 
                                        network_config['network']['network_critic_head'], 
                                        config['network']['network_critic_tail'])

        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.learning_rate)
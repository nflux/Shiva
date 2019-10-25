from .Agent import Agent
import networks.DDPGActor as actor
import networks.DDPGCritic as critic
import copy

class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        self.id = id

        # print("Look here: ", networks)



        self.actor = actor.DDPGActor(obs_dim, 
                                    action_dim, 
                                    networks['network_actor'])

        self.target_actor = copy.deepcopy(self.actor)

        self.critic = critic.DDPGCritic(obs_dim, 
                                        action_dim, 
                                        networks['network_critic_head'], 
                                        networks['network_critic_tail'])

        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.learning_rate)
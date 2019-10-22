from Agent import Agent

class DDPGAgent(Agent):
    def __init__(self, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DDPGAgent, self).__init__(obs_dim, action_dim, optimizer, learning_rate, config)

        self.actor = Network.DDPGActor(obs_dim, action_dim, config['network']['network_actor'])
        self.target_actor = self.actor

        self.critic = Network.DDPGCritic(obs_dim, action_dim, config['network']['network_critic_head'], config['network']['network_critic_tail'])
        self.target_critic = self.critic

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=learning_rate)
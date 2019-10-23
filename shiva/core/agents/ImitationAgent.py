from .Agent import Agent

class ImitationAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(ImitationAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim
        network_output = action_dim
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)
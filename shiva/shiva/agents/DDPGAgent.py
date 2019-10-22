from Agent import Agent

class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DDPGAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)
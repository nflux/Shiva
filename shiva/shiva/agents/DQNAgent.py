from Agent import Agent

class DQNAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, optimizer, learning_rate, config: dict):
        super(DQNAgent,self).__init__(id, obs_dim, action_dim, optimizer, learning_rate, config)
        network_input = obs_dim + action_dim
        network_output = 1
        self.policy = Network.initialize_network(network_input, network_output, config['network'])
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=learning_rate)
        
    def get_action(self, obs):
        '''
            This method iterates over all the possible actions to find the one with the highest Q value
        '''
        return self.find_best_action(self.policy, obs)

    def get_action_target(self, obs):
        '''
            Same as above but using the Target network
        '''
        return self.find_best_action(self.target_policy, obs)
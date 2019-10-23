from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
import copy

class DQNAgent(Agent):
    def __init__(self, agent_id, agent_config, net_config):
        super(DQNAgent,self).__init__(agent_id, agent_config)
        network_input = self.observation_space + self.action_space
        network_output = 1
        self.policy = DLN.DynamicLinearNetwork(network_input, network_output, net_config)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=self.learning_rate)
        
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
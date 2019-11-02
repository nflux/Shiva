from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
import helpers.networks_handler as nh
import copy
import torch.optim

class DQNAgent(Agent):
    def __init__(self, id, acs_space, obs_space, agent_config, net_config):
        super(DQNAgent,self).__init__(id, acs_space, obs_space, agent_config, net_config)
        self.id = id
        network_input = obs_space + acs_space
        network_output = 1

        print(net_config)
        
        self.policy = nh.DynamicLinearSequential(
                                        network_input, 
                                        network_output, 
                                        net_config['network']['layers'],
                                        nh.parse_functions(torch.nn, net_config['network']['activation_function']),
                                        net_config['network']['last_layer'],
                                        net_config['network']['output_function']
        )

        # self.policy = DLN.DynamicLinearNetwork(network_input, network_output, net_config)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.policy.parameters(), lr=agent_config['learning_rate'])
        
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
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
import copy
import torch.optim

class ImitationAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, agent_config,net_config):
        super(ImitationAgent,self).__init__(id, obs_dim, action_dim, agent_config,net_config)
        self.id = id
        network_input = obs_dim
        network_output = action_dim
        self.policy = DLN.DynamicLinearNetwork(network_input, network_output, net_config)
        self.optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.policy.parameters(), lr=agent_config['learning_rate'])

    def get_action(self, obs):
        '''
            This method iterates over all the possible actions to find the one with the highest Q value
        '''
        return self.find_best_action(self.policy, obs)

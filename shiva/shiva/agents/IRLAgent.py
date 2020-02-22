import copy
import torch.optim
from shiva.agents.Agent import Agent
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork


class IRLAgent(Agent):

    def __init__(self, id, acs_space, obs_space, agent_config, net_config):
        super(IRLAgent, self).__init__(id, acs_space, obs_space, agent_config, net_config)
        self.id = id
        network_input = obs_space + acs_space
        network_output = 1

        self.learning_rate = agent_config['learning_rate']

        self.reward = DynamicLinearNetwork(network_input, network_output, net_config['reward'])
        self.target_reward = copy.deepcopy(self.reward)

        self.optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.policy.parameters(),
                                                                                  lr=self.learning_rate)

    def get_reward(self, state, action,  step_n):
        '''
            Gets reward current reward function estimator
        '''
        return self.reward(state, action)

    def get_reward_target(self, state, action):
        '''
            Gets reward from target reward function estimator
            for Experimental Update
        '''
        return self.target_reward(state, action)

    def save(self, save_path, step):
        torch.save(self.reward, save_path + '/reward.pth')
        torch.save(self.target_reward, save_path + '/target_reward.pth')

    def __str__(self):
        return 'IRLAgent'

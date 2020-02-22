import copy
import torch.optim
import numpy as np

from shiva.agents.Agent import Agent
# from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork
import shiva.helpers.networks_handler as nh
from shiva.helpers.misc import action2one_hot_v


class IRLAgent(Agent):
    def __init__(self, id, acs_space, obs_space, agent_config, net_config):
        super(IRLAgent, self).__init__(id, acs_space, obs_space, agent_config, net_config)
        self.id = id
        network_input = obs_space + acs_space
        network_output = 1
        self.learning_rate = agent_config['learning_rate']

        self.policy = nh.DynamicLinearSequential(
            network_input,
            network_output,
            net_config['network']['layers'],
            nh.parse_functions(torch.nn, net_config['network']['activation_function']),
            net_config['network']['last_layer'],
            net_config['network']['output_function']
        )

        # self.policy = DynamicLinearNetwork(network_input, network_output, net_config)
        self.target_policy = copy.deepcopy(self.policy)
        self.optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.policy.parameters(),
                                                                                  lr=self.learning_rate)

    def get_reward(self, obs, step_n):
        '''
            This method iterates over all the possible actions to find the one with the highest Q value
        '''
        return self.find_best_action(self.policy, obs)

    def get_reward_target(self, obs):
        '''
            Same as above but using the Target network
        '''
        return self.find_best_action(self.target_policy, obs)

    def find_best_action(self, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = action2one_hot_v(self.acs_space, i)
            q_val = self.policy(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v
        return np.argmax(best_act).numpy()

    def get_reward(self, state, action) -> np.ndarray:
        '''
            With the probability epsilon we take the random action,te
            otherwise we use the network to obtain the best Q-value per each action
        '''

        reward = self.get_reward(state, action)

        return reward  # replay buffer store lists and env does np.argmax(action)

    def save(self, save_path, step):
        torch.save(self.policy, save_path + '/policy.pth')
        torch.save(self.target_policy, save_path + '/target_policy.pth')

    def __str__(self):
        return 'IRLAgent'
import copy
import torch.optim
import numpy as np
import random
import shiva.helpers.misc as misc

from shiva.agents.Agent import Agent
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork
import shiva.helpers.networks_handler as nh
from shiva.helpers.misc import action2one_hot

class DQNAgent(Agent):
    def __init__(self, id, acs_space, obs_space, agent_config, net_config):
        super(DQNAgent,self).__init__(id, acs_space, obs_space, agent_config, net_config)
        self.id = id
        network_input = obs_space + acs_space
        network_output = 1
        self.agent_config = agent_config
        if isinstance(agent_config['learning_rate'],list):
            self.learning_rate = random.uniform(agent_config['learning_rate'][0],agent_config['learning_rate'][1])
        else:
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
        self.optimizer = getattr(torch.optim, agent_config['optimizer_function'])(params=self.policy.parameters(), lr=self.learning_rate)

    def get_action(self, obs, step_n=0, evaluate=False):
        '''
            This method iterates over all the possible actions to find the one with the highest Q value
        '''
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs).float()
        if evaluate:
            return self.find_best_action(self.policy, obs)
        if step_n < self.exploration_steps or random.uniform(0, 1) < max(self.epsilon_end, self.epsilon_start - (step_n / self.epsilon_decay)):
            '''Random action or e-greedy exploration'''
            # check if obs is a batch!
            if len(obs.shape) > 1:
                # it's a batch operation!
                action = [self.get_random_action() for _ in range(obs.shape[0])]
                # print('random batch action')
            else:
                action = self.get_random_action()
                # print("random act")
        else:
            # print('action! {} {}'.format(obs, obs.shape))
            action = self.find_best_action(self.policy, obs)
        # print("From DQAgent Acs {}".format(action))
        return action

    def get_action_target(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs).float()
        return self.find_best_action(self.target_policy, obs)

    def get_random_action(self):
        action_idx = random.sample(range(self.acs_space), 1)[0]
        action = list(action2one_hot(self.acs_space, action_idx))
        return action

    def find_best_action(self, network, observation) -> np.ndarray:
        if len(observation.shape) > 1:
            '''This is for batch operation, as we need to find the highest Q for each obs'''
            return [self.find_best_action_from_tensor(network, ob) for ob in observation ]
        else:
            return self.find_best_action_from_tensor(network, observation)

    def find_best_action_from_tensor(self, network, obs_v) -> np.ndarray:
        '''
            Iterates over the action space to find the one with the highest Q value

            Input
                network         policy network to be used
                observation     observation from the environment

            Returns
                A one-hot encoded list
        '''
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space, i).to(self.device)
            q_val = network(torch.cat([obs_v.float(), act_v.float()], dim=-1))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v.tolist()
        return best_act

    def find_best_imitation_action(self, observation: np.ndarray) -> np.ndarray:

        obs_v = torch.tensor(observation).float().to(self.device)
        best_q, best_act_v = float('-inf'), torch.zeros(self.acs_space).to(self.device)
        for i in range(self.acs_space):
            act_v = misc.action2one_hot_v(self.acs_space,i)
            q_val = self.policy(torch.cat([obs_v, act_v.to(self.device)]))
            if q_val > best_q:
                best_q = q_val
                best_act_v = act_v
        best_act = best_act_v
        return np.argmax(best_act).numpy()

    def save(self, save_path, step):
        torch.save(self.policy, save_path + '/policy.pth')
        torch.save(self.target_policy, save_path + '/target_policy.pth')


    def copy_weights(self,evo_agent):
        print('Copying Weights')
        self.learning_rate = evo_agent.learning_rate
        self.optimizer = copy.deepcopy(evo_agent.optimizer)
        self.policy = copy.deepcopy(evo_agent.policy)
        self.target_policy = copy.deepcopy(evo_agent.target_policy)
        print('Copied Weights')

    def perturb_hyperparameters(self,perturb_factor):
        print('Pertubing HP')
        self.learning_rate = self.learning_rate * perturb_factor
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=self.learning_rate)
        print('Pertubed HP')

    def resample_hyperparameters(self):
        print('Resampling')
        self.learning_rate = random.uniform(self.agent_config['learning_rate'][0],self.agent_config['learning_rate'][1])
        self.optimizer = self.optimizer_function(params=self.policy.parameters(), lr=self.learning_rate)
        print('Resampled')

    def __str__(self):
        return '<DQNAgent(id={}, steps={}, episodes={})>'.format(self.id, self.step_count, self.done_count)

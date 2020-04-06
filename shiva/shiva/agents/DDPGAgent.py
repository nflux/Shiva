import numpy as np
import random
import torch
import copy
import pickle
from torch.distributions import Categorical
from torch.nn.functional import softmax
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.Agent import Agent
from shiva.utils import Noise as noise
from shiva.helpers.networks_helper import perturb_optimizer
from shiva.helpers.misc import action2one_hot, action2one_hot_v
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork


class DDPGAgent(Agent):
    def __init__(self, id:int, obs_space:int, acs_space:dict, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)
        try:
            self.seed = self.manual_seed
            torch.manual_seed(self.manual_seed)
            np.random.seed(self.manual_seed)
        except:
            self.seed = np.random.randint(0, 100)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.discrete = acs_space['discrete']
        self.continuous = acs_space['continuous']
        self.param = acs_space['discrete']

        if self.continuous == 0:
            self.action_space = 'discrete'
            self.actor_input = obs_space
            self.actor_output = self.discrete
            self.get_action = self.get_discrete_action
        elif self.discrete == 0:
            self.action_space = 'continuous'
            self.actor_input = obs_space
            self.actor_output = self.continuous
            self.get_action = self.get_continuous_action
        else:
            self.action_space = 'parametrized'
            self.actor_input = obs_space
            self.actor_output = self.discrete + self.param
            self.get_action = self.get_parameterized_action

        if hasattr(self, 'lr_range') and self.lr_range:
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
            self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
            self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        else:
            self.actor_learning_rate = self.agent_config['actor_learning_rate']
            self.critic_learning_rate = self.agent_config['critic_learning_rate']

        if 'MADDPG' not in str(self):
            self.critic_input_size = obs_space + self.actor_output

        self.ou_noise = noise.OUNoise(self.actor_output, self.noise_scale)
        self.instantiate_networks()

    def instantiate_networks(self):
        self.net_names = ['actor', 'target_actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer']

        self.actor = SoftMaxHeadDynamicLinearNetwork(self.actor_input, self.actor_output, self.param, self.networks_config['actor'])
        self.target_actor = copy.deepcopy(self.actor)
        '''If want to save memory on an MADDPG (not multicritic) run, put critic networks inside if statement'''
        self.critic = DynamicLinearNetwork(self.critic_input_size, 1, self.networks_config['critic'])
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

    def to_device(self, device):
        self.device = device
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)

    def get_discrete_action(self, observation, step_count, evaluate=False, one_hot=False, *args, **kwargs):
        self.ou_noise.set_scale(self.noise_scale)
        if evaluate:
# <<<<<<< HEAD
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            action = torch.from_numpy(action.cpu().numpy())
            '''???'''
            action = torch.abs(action)
            action = action / action.sum()
# =======
#             action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
#             self.ou_noise.set_scale(self.noise_scale)
#             action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
#             action = torch.abs(action)
#             action = action / action.sum()
#             # print("Agent Evaluate {}".format(action))
# >>>>>>> robocup-mpi-pbt
        else:
            if step_count < self.exploration_steps or self.is_e_greedy(step_count):
                action = np.array([np.random.uniform(0,1) for _ in range(self.actor_output)])
                action = torch.from_numpy(action + self.ou_noise.noise())
                action = softmax(action, dim=-1)
                # print("Random: {}".format(action))
            # elif np.random.uniform(0, 1) < self.epsilon:
            # elif self.is_e_greedy(step_count):
            #     action = np.array([np.random.uniform(0, 1) for _ in range(self.actor_output)])
            #     action = softmax(torch.from_numpy(action), dim=-1)
            else:
                action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
                action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
                action = torch.abs(action)
                action = action / action.sum()
                # print("Net: {}".format(action))
        if one_hot:
            action = action2one_hot(action.shape[0], torch.argmax(action).item())

        # until this point the action was a tensor, we are returning a python list - needs to be checked.
        return action.tolist()

    def get_continuous_action(self,observation, step_count, evaluate):
        raise NotImplemented
        # if self.evaluate:
        #     observation = torch.tensor(observation).float().to(self.device)
        #     action = self.actor(observation)
        # else:
        #     if step_count < self.exploration_steps:
        #         self.ou_noise.set_scale(self.exploration_noise)
        #         action = np.array([np.random.uniform(0, 1) for _ in range(self.actor_output)])
        #         action += self.ou_noise.noise()
        #         action = softmax(torch.from_numpy(action))
        #     else:
        #         observation = torch.tensor(observation).float().to(self.device)
        #         action = self.actor(observation)
        #         self.ou_noise.set_scale(self.training_noise)
        #         action += torch.tensor(self.ou_noise.noise()).float().to(self.device)
        #         action = action/action.sum()
        # return action.tolist()

    def get_parameterized_action(self, observation, step_count, evaluate=False):
        raise NotImplemented
        # if self.evaluate:
        #     observation = torch.tensor(observation).to(self.device)
        #     action = self.actor(observation.float())
        # else:
        #     # if step_count > self.exploration_steps:
        #     # else:
        #     observation = torch.tensor(observation).to(self.device)
        #     action = self.actor(observation.float())
        # # return action.tolist()
        # return action[0]

    def reset_noise(self):
        self.ou_noise.reset()

    def get_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float())
        return action[0]

    # def save(self, save_path, step):
    #     torch.save(self.actor, save_path + '/actor.pth')
    #     torch.save(self.target_actor, save_path + '/target_actor.pth')
    #     torch.save(self.critic, save_path + '/critic.pth')
    #     torch.save(self.target_critic, save_path + '/target_critic.pth')
    #     torch.save(self.actor_optimizer, save_path + '/actor_optimizer.pth')
    #     torch.save(self.critic_optimizer, save_path + '/critic_optimizer.pth')

    def copy_weights(self, evo_agent):
        self.actor_learning_rate = evo_agent.actor_learning_rate
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self.actor_optimizer = copy.deepcopy(evo_agent.actor_optimizer)
        self.critic_optimizer = copy.deepcopy(evo_agent.critic_optimizer)
        self.epsilon = evo_agent.epsilon
        self.noise_scale = evo_agent.noise_scale

        self.actor.load_state_dict(evo_agent.actor.to(self.device).state_dict())
        self.target_actor.load_state_dict(evo_agent.target_actor.to(self.device).state_dict())
        self.critic.load_state_dict(evo_agent.critic.to(self.device).state_dict())
        self.target_critic.load_state_dict(evo_agent.target_critic.to(self.device).state_dict())

    def perturb_hyperparameters(self, perturb_factor):
        self.actor_learning_rate = self.actor_learning_rate * perturb_factor
        self.critic_learning_rate = self.critic_learning_rate * perturb_factor
        # for param_group in self.actor_optimizer.param_groups:
        #     param_group['lr'] = self.actor_learning_rate
        # for param_group in self.critic_optimizer.param_groups:
        #     param_group['lr'] = self.critic_learning_rate
        self.actor_optimizer = perturb_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
        self.critic_optimizer = perturb_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})
        self.epsilon *= perturb_factor
        self.noise_scale *= perturb_factor

    def resample_hyperparameters(self):
        self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        # for param_group in self.actor_optimizer.param_groups:
        #     param_group['lr'] = self.actor_learning_rate
        # for param_group in self.critic_optimizer.param_groups:
        #     param_group['lr'] = self.critic_learning_rate
        self.actor_optimizer = perturb_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
        self.critic_optimizer = perturb_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})
        self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])

    def is_e_greedy(self, step_count=None):
        '''E-greedy with linear decay'''
        if step_count is None:
            step_count = self.step_count
        r = random.uniform(0, 1)
        ceiling = max(self.epsilon, self.epsilon_start - (step_count / self.epsilon_decay))
        is_random = r < ceiling
        return is_random

    def get_metrics(self):
        '''Used for evolution metric'''
        return [
            ('Agent/{}/Actor_Learning_Rate'.format(self.role), self.actor_learning_rate),
            ('Agent/{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
        ]

    def get_module_and_classname(self):
        return ('shiva.agents', 'DDPGAgent.DDPGAgent')

    def __str__(self):
        return '<DDPGAgent(id={}, role={}, steps={}, episodes={}, num_updates={}, device={})>'.format(self.id, self.role, self.step_count, self.done_count, self.num_updates, self.device)

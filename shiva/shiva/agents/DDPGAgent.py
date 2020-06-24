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
from shiva.helpers.networks_helper import mod_optimizer
from shiva.helpers.misc import action2one_hot, action2one_hot_v
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork


class DDPGAgent(Agent):
    def __init__(self, id:int, obs_space:int, acs_space:dict, configs):
        super(DDPGAgent, self).__init__(id, obs_space, acs_space, configs)
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
            if not hasattr(self, 'actions_range'):
                self.actions_range = [-1, 1]
        else:
            self.action_space = 'parametrized'
            self.actor_input = obs_space
            self.actor_output = self.discrete + self.param
            self.get_action = self.get_parameterized_action

        self.instantiate_networks()

    def instantiate_networks(self):
        self.hps = ['actor_learning_rate', 'critic_learning_rate']
        self.hps += ['epsilon', 'noise_scale']
        self.hps += ['epsilon_start', 'epsilon_end', 'epsilon_episodes', 'noise_start', 'noise_end', 'noise_episodes']

        self.ou_noise = noise.OUNoise(self.actor_output, self.noise_scale)

        if hasattr(self, 'lr_range') and self.lr_range:
            self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
            self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])
            self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
            self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        else:
            self.epsilon = self.epsilon_start
            self.noise_scale = self.noise_start
            self.actor_learning_rate = self.agent_config['actor_learning_rate']
            self.critic_learning_rate = self.agent_config['critic_learning_rate']

        if 'MADDPG' not in str(self):
            self.critic_input_size = obs_space + self.actor_output

        self.net_names = ['actor', 'target_actor', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer']

        if self.action_space == 'continuous':
            self.actor = DynamicLinearNetwork(self.actor_input, self.actor_output, self.networks_config['actor'])
        elif self.action_space == 'discrete':
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
        if evaluate:
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            # print("Agent Evaluate {}".format(action))
        else:
            if self.is_exploring(step_count) or self.is_e_greedy(step_count):
                action = np.array([np.random.uniform(0, 1) for _ in range(self.actor_output)])
                action = torch.from_numpy(action + self.ou_noise.noise())
                action = softmax(action, dim=-1)
                # print("Random: {}".format(action))
            else:
                print(f"Observation: {observation}")
                action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
                print(f"Before Noise: {action}")
                action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
                # print(f"Action Before ABS: {action}")
                action = torch.abs(action)
                # print(f"Action After ABS: {action}")
                action = action / action.sum()
                # print(f"Action After Normalization: {action}")
        if one_hot:
            action = action2one_hot(action.shape[0], torch.argmax(action).item())

        # until this point the action was a tensor, we are returning a python list - needs to be checked.
        return action.tolist()

    def get_continuous_action(self, observation, step_count, evaluate=False, *args, **kwargs):
        if evaluate:
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            # action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
        else:
            if self.is_exploring(step_count) or self.is_e_greedy(step_count):
                action = np.array([np.random.uniform(*self.actions_range) for _ in range(self.actor_output)])
                # action = torch.from_numpy(action + self.ou_noise.noise())
                # action = softmax(action, dim=-1)
                # self.log(f"** Random action {action.tolist()}", verbose_level=1)
            else:
                action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
                action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
                action = torch.clamp(action, min=self.actions_range[0], max=self.actions_range[1])
                self.log(f"Network action {action.tolist()}", verbose_level=1)
        return action.tolist()

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

    def copy_hyperparameters(self, evo_agent):
        self.actor_learning_rate = evo_agent.actor_learning_rate
        self.critic_learning_rate = evo_agent.critic_learning_rate
        self.actor_optimizer = mod_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
        self.critic_optimizer = mod_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})
        self.epsilon = evo_agent.epsilon
        self.noise_scale = evo_agent.noise_scale

    def copy_weights(self, evo_agent):
        self.actor.load_state_dict(evo_agent.actor.to(self.device).state_dict())
        self.target_actor.load_state_dict(evo_agent.target_actor.to(self.device).state_dict())
        self.critic.load_state_dict(evo_agent.critic.to(self.device).state_dict())
        self.target_critic.load_state_dict(evo_agent.target_critic.to(self.device).state_dict())
        self.actor_optimizer.load_state_dict(evo_agent.actor_optimizer.state_dict())
        self.critic_optimizer.load_state_dict(evo_agent.critic_optimizer.state_dict())

    def perturb_hyperparameters(self, perturb_factor):
        self.actor_learning_rate *= perturb_factor
        self.critic_learning_rate *= perturb_factor
        self.actor_optimizer = mod_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
        self.critic_optimizer = mod_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})
        self.epsilon *= perturb_factor
        self.noise_scale *= perturb_factor

    def resample_hyperparameters(self):
        self.actor_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self.critic_learning_rate = np.random.uniform(self.agent_config['lr_uniform'][0], self.agent_config['lr_uniform'][1]) / np.random.choice(self.agent_config['lr_factors'])
        self.actor_optimizer = mod_optimizer(self.actor_optimizer, {'lr': self.actor_learning_rate})
        self.critic_optimizer = mod_optimizer(self.critic_optimizer, {'lr': self.critic_learning_rate})
        self.epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1])
        self.noise_scale = np.random.uniform(self.ou_range[0], self.ou_range[1])

    def is_e_greedy(self, step_count=None):
        if step_count is None:
            step_count = self.step_count
        if step_count > self.exploration_steps:
            step_count = step_count - self.exploration_steps # don't count explorations steps
            return random.uniform(0, 1) < self.epsilon
        else:
            return True

    def update_epsilon_scale(self, done_count=None):
        '''To be called by the Learner before saving'''
        if done_count is None:
            done_count = self.done_count
        if self.is_exploring():
            return self.epsilon_start
        self.epsilon = self._get_epsilon_scale(done_count)

    def _get_epsilon_scale(self, done_count=None):
        if done_count is None:
            done_count = self.done_count
        avr_exploration_episodes = self.exploration_steps / (self.step_count / self.done_count)
        return max(self.epsilon_end, self.decay_value(self.epsilon_start, self.epsilon_episodes, (done_count - avr_exploration_episodes), degree=self.epsilon_decay_degree))

    def update_noise_scale(self, done_count=None):
        '''To be called by the Learner before saving'''
        if done_count is None:
            done_count = self.done_count
        self.noise_scale = self._get_noise_scale(done_count)
        self.ou_noise.set_scale(self.noise_scale)

    def _get_noise_scale(self, done_count=None):
        if done_count is None:
            done_count = self.done_count
        if self.is_exploring():
            return self.noise_start
        avr_exploration_episodes = self.exploration_steps / (self.step_count / self.done_count)
        return max(self.noise_end, self.decay_value(self.noise_start, self.noise_episodes, (done_count - (avr_exploration_episodes)), degree=self.noise_decay_degree))

    def reset_noise(self):
        self.ou_noise.reset()

    def decay_value(self, start, decay_end_step, current_step_count, degree=1):
        return start - start * ((current_step_count / decay_end_step) ** degree)

    def is_exploring(self, current_step_count=None):
        if hasattr(self, 'exploration_episodes'):
            if current_step_count is None:
                current_step_count = self.done_count
            _threshold = self.exploration_episodes
        else:
            if current_step_count is None:
                current_step_count = self.step_count
            _threshold = self.exploration_steps
        return current_step_count < _threshold

    def get_metrics(self):
        '''Used for evolution metric'''
        return [
            ('{}/Actor_Learning_Rate'.format(self.role), self.actor_learning_rate),
            ('{}/Critic_Learning_Rate'.format(self.role), self.critic_learning_rate),
            ('{}/Epsilon'.format(self.role), self.epsilon),
            ('{}/Noise_Scale'.format(self.role), self.noise_scale),
        ]

    def get_module_and_classname(self):
        return ('shiva.agents', 'DDPGAgent.DDPGAgent')

    def __str__(self):
        return f"'<DDPGAgent(id={self.id}, role={self.role}, S/E/U={self.step_count}/{self.done_count}/{self.num_updates}, device={self.device})>'"
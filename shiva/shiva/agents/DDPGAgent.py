import numpy as np
import torch
import copy
import pickle
from torch.distributions import Categorical
from torch.nn.functional import softmax
from shiva.helpers.calc_helper import np_softmax
from shiva.agents.Agent import Agent
from shiva.utils import Noise as noise
from shiva.helpers.misc import action2one_hot, action2one_hot_v
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork

class DDPGAgent(Agent):
    def __init__(self, id:int, obs_space:int, acs_space:dict, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_space, acs_space, agent_config, networks)
        try:
            torch.manual_seed(self.manual_seed)
            np.random.seed(self.manual_seed)
        except:
            torch.manual_seed(5)
            np.random.seed(5)

        '''
        
            Maybe do something like

        
        '''
        self.discrete = acs_space['discrete']
        self.continuous = acs_space['continuous']
        self.param = acs_space['discrete']
        self.acs_space = acs_space['acs_space']

        # print(self.discrete, self.continuous, self.param, self.acs_space)
        # print(obs_space)
        
        if self.continuous == 0:
            self.action_space = 'discrete'
            actor_input = obs_space
            actor_output = self.discrete
            self.get_action = self.get_discrete_action
        elif self.discrete == 0:
            self.action_space = 'continuous'
            actor_input = obs_space
            actor_output = self.continuous
            self.get_action = self.get_continuous_action
        else:
            self.action_space = 'parametrized'
            actor_input = obs_space
            actor_output = self.discrete + self.param
            self.get_action = self.get_parameterized_action

        self.actor = SoftMaxHeadDynamicLinearNetwork(actor_input, actor_output, self.param, networks['actor'])
        self.target_actor = copy.deepcopy(self.actor)
        if not hasattr(self, 'critic_input_size'):
            self.critic_input_size = obs_space + self.acs_space

        self.critic = DynamicLinearNetwork(self.critic_input_size, 1, networks['critic'])
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        try:
            self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)
        except:
            pass

        self.ou_noise = noise.OUNoise(self.acs_space, self.exploration_noise)

    # def _get_action(self, observation, step_count, one_hot=True, evaluate=None):
    #     if evaluate is not None:
    #         self.evaluate = evaluate
    #     if self.action_space == 'discrete':
    #         return self.get_discrete_action(observation, step_count, one_hot, self.evaluate)
    #     elif self.action_space == 'continuous':
    #         return self.get_continuous_action(observation, step_count, self.evaluate)
    #     elif self.action_space == 'parameterized':
    #         assert "DDPG Parametrized NotImplemented"
    #         pass
    #         return self.get_parameterized_action(observation, self.evaluate)

    def get_discrete_action(self, observation, step_count, one_hot=True, evaluate=False):
        if self.evaluate:
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            # print("Agent Evaluate {}".format(action))
        else:
            if step_count < self.exploration_steps:
                self.ou_noise.set_scale(self.exploration_noise)
                action = np.array([np.random.uniform(0,1) for _ in range(self.acs_space)])
                action = torch.from_numpy(action + self.ou_noise.noise())
                action = softmax(action, dim=-1)
                # print("Random: {}".format(action))
            else:
                self.ou_noise.set_scale(self.training_noise)
                action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
                action = torch.from_numpy(action.cpu().numpy() + self.ou_noise.noise())
                action = torch.abs(action)
                action = action / action.sum()
                # print("Net: {}".format(action))
        if one_hot:
            action = action2one_hot(action.shape[0], torch.argmax(action).item())
        return action.tolist()

    def get_continuous_action(self,observation, step_count, evaluate):
        if self.evaluate:
            observation = torch.tensor(observation).float().to(self.device)
            action = self.actor(observation)
        else:
            if step_count < self.exploration_steps:
                self.ou_noise.set_scale(self.exploration_noise)
                action = np.array([np.random.uniform(0,1) for _ in range(self.acs_space)])
                action += self.ou_noise.noise()
                action = softmax(torch.from_numpy(action))

            else:
                observation = torch.tensor(observation).float().to(self.device)
                action = self.actor(observation)
                self.ou_noise.set_scale(self.training_noise)
                action += torch.tensor(self.ou_noise.noise()).float().to(self.device)
                action = action/action.sum()

        return action.tolist()

    def get_parameterized_action(self, observation, step_count, evaluate=False):
        if self.evaluate:
            observation = torch.tensor(observation).to(self.device)
            action = self.actor(observation.float())           
        else:
            # if step_count > self.exploration_steps:
            # else:
            observation = torch.tensor(observation).to(self.device)
            action = self.actor(observation.float())
        # return action.tolist()

        return action[0]

    def reset_noise(self):
        self.ou_noise.reset()

    def get_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float())
        return action[0]

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')
        torch.save(self.actor_optimizer, save_path + '/actor_optimizer.pth')
        torch.save(self.critic_optimizer, save_path + '/critic_optimizer.pth')

    # def load(self,save_path):
    #     self.actor.load_state_dict(torch.load(save_path + 'actor.pth'))
    #     self.target_actor.load_state_dict(torch.load(save_path + 'target_actor.pth'))
    #     self.critic.load_state_dict(torch.load(save_path + 'critic.pth'))
    #     self.target_critic.load_state_dict(torch.load(save_path + 'target_critic.pth'))
    #     self.actor_optimizer.load_state_dict(torch.load(save_path + 'actor_optimizer.pth'))
    #     self.critic_optimizer.load_state_dict(torch.load(save_path + 'critic_optimizer.pth'))
        
    def __str__(self):
        return '<DDPGAgent(id={}, role={}, steps={}, episodes={}, num_updates={})>'.format(self.id, self.role, self.step_count, self.done_count, self.num_updates)
        
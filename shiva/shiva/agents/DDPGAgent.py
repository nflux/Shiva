import numpy as np
import torch
import copy
from torch.distributions import Categorical
from shiva.agents.Agent import Agent
from shiva.utils import Noise as noise
from shiva.helpers.misc import action2one_hot
from shiva.networks.DynamicLinearNetwork import DynamicLinearNetwork, SoftMaxHeadDynamicLinearNetwork

class DDPGAgent(Agent):
    def __init__(self, id, obs_dim, action_dim, param_ix, agent_config: dict, networks: dict):
        super(DDPGAgent, self).__init__(id, obs_dim, action_dim, agent_config, networks)
        try:
            torch.manual_seed(self.manual_seed)
            np.random.seed(self.manual_seed)
        except:
            torch.manual_seed(5)
            np.random.seed(5)

        self.id = id

        self.actor = SoftMaxHeadDynamicLinearNetwork(obs_dim, action_dim, param_ix, networks['actor'])
        self.target_actor = copy.deepcopy(self.actor)

        self.critic = DynamicLinearNetwork(obs_dim + action_dim, 1, networks['critic'])
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = self.optimizer_function(params=self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = self.optimizer_function(params=self.critic.parameters(), lr=self.critic_learning_rate)

        self.ou_noise = noise.OUNoise(action_dim, self.exploration_noise)
        self.acs_discrete = action_dim

    def get_action(self, observation, step_count, evaluate=False):
        if self.action_space == 'discrete':
            return self.get_discrete_action(observation, step_count, evaluate)
        elif self.action_space == 'continuous':
            return self.get_continuous_action(observation, step_count, evaluate)
        elif self.action_space == 'parameterized':
            pass
            return self.get_parameterized_action(observation, evaluate)

    def get_discrete_action(self, observation, step_count, evaluate):
        if evaluate:
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            action = torch.argmax(action)
            action = action2one_hot(self.acs_discrete, action.item())
        else:
            action = self.actor(torch.tensor(observation).to(self.device).float()).detach()
            action = Categorical(action).sample()
            action = action2one_hot(self.acs_discrete, action.item())
        return action.tolist()
            
    def get_continuous_action(self,observation, step_count, evaluate):
        if evaluate:
            observation = torch.tensor(observation).float().to(self.device)
            action = self.actor(observation)
        else:
            observation = torch.tensor(observation).float().to(self.device)
            action = self.actor(observation)
            self.ou_noise.set_scale(self.exploration_noise)
            action += torch.tensor(self.ou_noise.noise()).float().to(self.device)
            action = np.clip(action.cpu().detach().numpy(),-1,1)
        return action.tolist()

    def get_parameterized_action(self, observation, step_count, evaluate):
        if evaluate or step_count > self.exploration_steps:
            pass
        else:
            pass
        pass
        return action.tolist()

    def find_best_imitation_action(self, observation: np.ndarray) -> np.ndarray:
        observation = torch.tensor(observation).to(self.device)
        action = self.actor(observation.float()).cpu().data.numpy()
        action = np.clip(action, -1,1)
        # print('actor action shape', action.shape)
        return action[0]

    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.target_actor, save_path + '/target_actor.pth')
        torch.save(self.critic, save_path + '/critic.pth')
        torch.save(self.target_critic, save_path + '/target_critic.pth')

    def save_agent(self, save_path, step):
        torch.save(self.actor.state_dict(), save_path + '/actor.pth')
        torch.save(self.target_actor.state_dict(), save_path + '/target_actor.pth')
        torch.save(self.critic.state_dict(), save_path + '/critic.pth')
        torch.save(self.target_critic.state_dict(), save_path + '/target_critic.pth')

    def load(self, save_path):
        self.actor.load_state_dict(torch.load(save_path + '/actor.pth'))
        self.target_actor.load_state_dict(torch.load(save_path + '/target_actor.pth'))
        self.critic.load_state_dict(torch.load(save_path + '/critic.pth'))
        self.target_critic.load_state_dict(torch.load(save_path + '/target_critic.pth'))   
        
    def __str__(self):
        return 'DDPGAgent'


    # def save_agent(self, save_path,step):

    #     torch.save({
    #         'actor': self.actor.state_dict(),
    #         'critic': self.critic.state_dict(),
    #         'target_actor' : self.target_actor.state_dict(),
    #         'target_critic' : self.target_critic.state_dict(),
    #         'agent' : self
    #     }, save_path + '/agent.pth')

    # def load(self,save_path):
    #     print(save_path)
    #     model = torch.load(save_path + '/agent.pth')
    #     self.agent = model['agent']
    #     self.target_critic.load_state_dict(model['target_critic'])
    #     self.target_actor.load_state_dict(model['target_actor'])
    #     self.critic.load_state_dict(model['critic'])
    #     self.actor.load_state_dict( model['actor'])     

    # # Gets actions with a linearly decreasing e greedy strat
    # def get_action(self, observation) -> np.ndarray: # maybe a torch.tensor
    #     # print('get action')
    #     # if step_count < self.exploration_steps:
    #     if True:
    #         self.ou_noise.set_scale(self.exploration_noise)
    #         action = np.array([np.random.uniform(0,1) for _ in range(self.discrete+self.param)])
    #         action += self.ou_noise.noise()
    #         action = np.concatenate([ np_softmax(action[:self.discrete]), action[self.discrete:] ])
    #         self.action = np.clip(action, -1, 1)
    #         # print('random action shape', action[:self.acs_space['discrete']].sum(), action.shape)
    #         return self.action

    #     else:

    #         self.ou_noise.set_scale(self.training_noise)
    #         observation = torch.tensor([observation]).to(self.device)
    #         action = agent.get_action(observation.float()).cpu().data.numpy()

    #         # print("Network Output (after softmax):", action)
    #         # input()

    #         # useful for debugging
    #         # if step_count % 100 == 0:
    #             # print(action)

    #         # action += self.ou_noise.noise()
    #         # action = np.clip(action, -1,1)
    #         # print('actor action shape', action.shape)

    #         size = len(action.shape)
    #         if size == 3:
    #             self.action = action[0, 0]
    #             return action[0, 0]
    #         elif size == 2:
    #             self.action = action[0]
    #             return action[0]
    #         else:    
    #             self.action = action
    #             return action

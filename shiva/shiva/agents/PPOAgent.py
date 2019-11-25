from .Agent import Agent
from .Agent import Agent
import networks.DynamicLinearNetwork as DLN
from torch.distributions import Categorical
import utils.Noise as noise
import copy
import torch
import numpy as np
import helpers.misc as misc

class PPOAgent(Agent):
    def __init__(self, id, obs_dim, acs_discrete, acs_continuous, agent_config,net_config):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.acs_discrete = acs_discrete
        self.acs_continuous = acs_continuous
        self.ou_noise = noise.OUNoise(acs_discrete, 0.9)
        if (acs_discrete != None) and (acs_continuous != None):
            self.network_output = self.acs_discrete + self.acs_continuous
        elif (self.acs_discrete == None):
            self.network_output = self.acs_continuous
        else:
            self.network_output = self.acs_discrete
        self.id = id
        self.network_input = obs_dim

        self.actor = DLN.DynamicLinearNetwork(self.network_input, self.network_output, net_config['actor'])
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = DLN.DynamicLinearNetwork(self.network_input, 1, net_config['critic'])
        self.target_critic = copy.deepcopy(self.critic)



        self.actor_optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.actor.parameters(), lr=agent_config['learning_rate'])
        self.critic_optimizer = getattr(torch.optim,agent_config['optimizer_function'])(params=self.critic.parameters(), lr=agent_config['learning_rate'])


    def get_action(self,observation):
        #retrieve the action given an observation
        full_action = self.target_actor(torch.tensor(observation).float()).detach().numpy()
        if(len(full_action.shape) > 1):
            full_action = full_action.reshape(len(full_action[0]))

        #Check if there is a discrete component to the action tensor
        if (self.acs_discrete != None):
            #If there is, extract the discrete component
            discrete_action = full_action[: self.acs_discrete]
            #discrete_action += self.ou_noise.noise()
            sampled_choice = np.zeros(len(discrete_action))
            #discrete_action = discrete_action.reshape((1,len(discrete_action)))
            sampled_action_1 = np.random.choice([0,1],1,p=discrete_action)
            #sampled_action = np.where(discrete_action == sampled_action_1)
            #sampled_choice[sampled_action[0]] = 1.0
            #discrete_action = sampled_choice.tolist()

            for i in range(len(discrete_action)):
                if i == sampled_action_1:
                    discrete_action[i] = 1
                else:
                    discrete_action[i] = 0

            #Recombine the chosen discrete action with the continuous action values
            full_action = np.concatenate([discrete_action,full_action[self.acs_discrete:]])

            return full_action.tolist()
        else:
            return full_action

    '''def get_action(self,observation):
        observation = torch.tensor(observation).clone().detach().float().requires_grad_(True)
        action_probs = self.target_actor(observation)
        #action_probs = self.actor(torch.from_numpy(observation).float()).detach().requires_grad_(True)
        dist = Categorical(action_probs)
        action = dist.sample()
        if action.dim() > 0:
            action_one_hot = [None] * len(observation)
            for i in range(len(action_one_hot)):
                action_one_hot[i] = misc.action2one_hot(self.acs_discrete,action[i].item()).tolist()
            return action_one_hot
        else:
            action = misc.action2one_hot(self.acs_discrete,action.item()).tolist()
            return action'''


    def save(self, save_path, step):
        torch.save(self.actor, save_path + '/actor.pth')
        torch.save(self.critic, save_path +'/critic.pth')

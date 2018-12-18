from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch
from .networks import MLPNetwork_Actor,MLPNetwork_Critic,I2A_Network
import torch.nn.functional as F
from .i2a import *
from .misc import hard_update, gumbel_softmax, onehot_from_logits,processor
from .noise import OUNoise
import numpy as np
class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, maddpg=object, hidden_dim=64,
                 a_lr=0.001, c_lr=0.001, discrete_action=True,n_atoms = 51,vmax=10,vmin=-10,delta=20.0/50,D4PG=True,TD3=False,
                I2A = False,EM_lr=0.001,world_status_dim = 6,rollout_steps = 5,LSTM_hidden=64,
                 device='cpu',imagination_policy_branch=True, critic_mod_both = False, critic_mod_act = False, critic_mod_obs = False): 
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        # NOTE: Only works for m vs m
        if critic_mod_both:
            num_in_critic = num_in_critic + ((num_in_pol + num_out_pol) * maddpg.nagents_team)
        elif critic_mod_act:
            num_in_critic = num_in_critic + (num_out_pol * maddpg.nagents_team)
        elif critic_mod_obs:
            num_in_critic = num_in_critic + (num_in_pol * maddpg.nagents_team)

        self.norm_in = False
        self.counter = 0
        self.updating_actor = False
        self.maddpg = maddpg
        self.param_dim = 5
        self.action_dim = 3
        self.imagination_policy_branch = imagination_policy_branch
        self.device = device
        self.n_actions = 1 + imagination_policy_branch# number of imagination branches
        self.delta = (float(vmax)-vmin)/(n_atoms-1)
        # D4PG
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.world_status_dim = world_status_dim
        self.num_out_obs_EM = num_in_critic - num_out_pol # obs + actions - actions  = obs head, (this does not include the ws head that is handled internally by network)
        # print('Num of maddpg agents and num out', str(maddpg.nagents_team), str(num_out_pol), str(self.num_out_obs_EM))
        # self.num_in_EM = num_in_critic - (self.num_out_obs_EM * (maddpg.nagents_team - 1)) # obs + actions 

        self.num_total_out_EM = maddpg.num_out_EM + self.world_status_dim + 1
        # EM for I2A
        self.EM = EnvironmentModel(maddpg.num_in_EM,maddpg.num_out_EM,hidden_dim=hidden_dim,
                                  norm_in=False,agent=self)
        # policy prime for I2A
        self.policy_prime = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 discrete_action=discrete_action, 
                                 norm_in= self.norm_in,agent=self)

        # Stores a policy such as Helios to be used as one of the branches in imagination
        self.imagination_policy = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 discrete_action=discrete_action, 
                                 norm_in= self.norm_in,agent=self)
       
        self.policy = I2A_Network(num_in_pol, num_out_pol, self.num_total_out_EM,
                          hidden_dim=hidden_dim,
                          discrete_action=discrete_action,
                          norm_in= self.norm_in,agent=self,I2A=I2A,rollout_steps=rollout_steps,
                          EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                  LSTM_hidden=LSTM_hidden,maddpg=maddpg)
        
        self.target_policy = I2A_Network(num_in_pol, num_out_pol,self.num_total_out_EM,
                                 hidden_dim=hidden_dim,
                                 discrete_action=discrete_action,
                                 norm_in= self.norm_in,agent=self,I2A=I2A,rollout_steps=rollout_steps,
                                 EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                         LSTM_hidden=LSTM_hidden,maddpg=maddpg)
        
        self.critic = MLPNetwork_Critic(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 norm_in= self.norm_in,agent=self,D4PG=D4PG,TD3=TD3)

        self.target_critic = MLPNetwork_Critic(num_in_critic, 1,
                                               hidden_dim=hidden_dim,
                                               norm_in= self.norm_in,agent=self,D4PG=D4PG,TD3=TD3)

        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.EM_lr = EM_lr
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=a_lr, weight_decay =0)
        self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=a_lr, weight_decay =0)
        self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=a_lr, weight_decay =0)
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=c_lr, weight_decay=0)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=c_lr, weight_decay=0)
        self.EM_optimizer = Adam(self.EM.parameters(), lr=EM_lr)

        if not discrete_action: # input to OUNoise is size of param space # TODO change OUNoise param to # params
            self.exploration = OUNoise(self.param_dim) # hard coded
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        
        action = self.policy(obs)
        #if self.counter % 100 == 0:
            #print(action)
        self.counter +=1
        #print(action)
        # mixed disc/cont
        if explore:     
            a = action[0,:self.action_dim].view(1,self.action_dim)
            p = (action[0,self.action_dim:].view(1,self.param_dim) + Variable(processor(Tensor(self.exploration.noise()),device=self.device),requires_grad=False)) # get noisey params (OU)
            self.exploration.reset()
            action = torch.cat((a,p),1) 
        #print(action)
        return action
            
        '''if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action
'''
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_prime': self.policy_prime.state_dict(),
                'imagination_policy': self.imagination_policy.state_dict(),
                'EM': self.EM.state_dict(),
                
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'policy_prime_optimizer': self.policy_prime_optimizer.state_dict(),
                'imagination_policy_optimizer': self.imagination_policy_optimizer.state_dict(),
                'EM_optimizer': self.EM_optimizer.state_dict(),                
               }

    #Used to get just the actor weights and params.
    def get_actor_params(self):
        return {'policy': self.policy.state_dict()}

    #Used to get just the critic weights and params.
    def get_critic_params(self):
        return {'critic': self.critic.state_dict()}

    def load_params(self, params):
        if self.device == 'cuda':
            dev = torch.device("cuda")
        else:
            dev = torch.device('cpu')
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_prime.load_state_dict(params['policy_prime'])
        self.imagination_policy.load_state_dict(params['imagination_policy'])
        self.EM.load_state_dict(params['EM'])


        self.policy.to(dev)
        self.critic.to(dev)
        self.target_policy.to(dev)
        self.target_critic.to(dev)
        self.policy_prime.to(dev)
        self.imagination_policy.to(dev)
        self.EM.to(dev)
      
        
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)
        self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=self.a_lr, weight_decay =0)
        self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=self.a_lr, weight_decay =0)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)
        self.EM_optimizer = Adam(self.EM.parameters(), lr=self.EM_lr)
        
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        self.policy_prime_optimizer.load_state_dict(params['policy_prime_optimizer'])
        self.imagination_policy_optimizer.load_state_dict(params['imagination_policy_optimizer'])
        self.EM_optimizer.load_state_dict(params['EM_optimizer'])

        
    def load_policy_params(self, params):
        if self.device == 'cuda':
            dev = torch.device("cuda")
        else:
            dev = torch.device('cpu')
        self.policy.load_state_dict(params['policy'])
        self.policy_prime.load_state_dict(params['policy_prime'])
        self.imagination_policy.load_state_dict(params['imagination_policy'])
        self.EM.load_state_dict(params['EM'])


        self.policy.to(dev)
        self.policy_prime.to(dev)
        self.imagination_policy.to(dev)
        self.EM.to(dev)
      
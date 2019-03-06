from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch
from .networks import MLPNetwork_Actor,MLPNetwork_Critic,LSTM_Network,I2A_Network,LSTMNetwork_Critic
import torch.nn.functional as F
from .i2a import *
from .misc import hard_update, gumbel_softmax, onehot_from_logits,processor,e_greedy_bool
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
                device='cpu',imagination_policy_branch=True, critic_mod_both = False, critic_mod_act = False, critic_mod_obs = False, 
                LSTM=False, seq_length=20, hidden_dim_lstm=256): 
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.I2A = I2A
        self.norm_in = False
        self.counter = 0
        self.updating_actor = False
        self.maddpg = maddpg
        self.batch_size = maddpg.batch_size
        self.param_dim = 5
        self.action_dim = 3
        self.imagination_policy_branch = imagination_policy_branch
        self.device = device
        self.n_branches = 1 + imagination_policy_branch# number of imagination branches
        self.delta = (float(vmax)-vmin)/(n_atoms-1)
        # D4PG
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.world_status_dim = world_status_dim
        self.LSTM = LSTM

        I2A_num_in_pol = num_in_pol
        self.hidden_dim_lstm = hidden_dim_lstm

        self.num_total_out_EM = maddpg.num_out_EM + self.world_status_dim + 1
        # EM for I2A
        if I2A:
            self.EM = EnvironmentModel(maddpg.num_in_EM,maddpg.num_out_EM,hidden_dim=hidden_dim,
                                  norm_in=self.norm_in,agent=self,maddpg=maddpg)
            # Stores a policy such as Helios to be used num_out_pol*env.num_TAas one of the branches in imagination
            self.imagination_policy = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                    hidden_dim=hidden_dim,
                                    discrete_action=discrete_action, 
                                    norm_in= self.norm_in,agent=self,maddpg=maddpg)
        # policy prime for I2A
            self.policy_prime = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                    hidden_dim=hidden_dim,
                                    discrete_action=discrete_action, 
                                    norm_in= self.norm_in,agent=self,maddpg=maddpg)
        else:
            self.EM = None
            self.imagination_policy = None
            self.policy_prime = None
        
 
        self.policy = I2A_Network(I2A_num_in_pol, num_out_pol, self.num_total_out_EM,
                        hidden_dim=hidden_dim,
                        discrete_action=discrete_action,
                        norm_in= self.norm_in,agent=self,I2A=I2A,rollout_steps=rollout_steps,
                        EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                LSTM_hidden=LSTM_hidden,maddpg=maddpg)

        if torch.cuda.device_count() > 1 and maddpg.data_parallel:
            self.policy = nn.DataParallel(self.policy)

        #self.policy.share_memory()
        if not maddpg.only_policy:

            self.target_policy = I2A_Network(I2A_num_in_pol, num_out_pol,self.num_total_out_EM,
                                    hidden_dim=hidden_dim,
                                    discrete_action=discrete_action,
                                    norm_in= self.norm_in,agent=self,I2A=I2A,rollout_steps=rollout_steps,
                                    EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                            LSTM_hidden=LSTM_hidden,maddpg=maddpg)
            if torch.cuda.device_count() > 1 and maddpg.data_parallel:
                self.target_policy = nn.DataParallel(self.target_policy)

        if not maddpg.only_policy:
            if LSTM:
                self.critic = LSTMNetwork_Critic(num_in_critic, 1,
                                            hidden_dim=hidden_dim, agent=self, n_atoms=n_atoms, D4PG=D4PG,TD3=TD3,maddpg=maddpg)
                self.target_critic = LSTMNetwork_Critic(num_in_critic, 1,
                                            hidden_dim=hidden_dim, agent=self, n_atoms=n_atoms, D4PG=D4PG,TD3=TD3,maddpg=maddpg)
            else:
                self.critic = MLPNetwork_Critic(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        norm_in= self.norm_in,agent=self,n_atoms=n_atoms,D4PG=D4PG,TD3=TD3,maddpg=maddpg)

                self.target_critic = MLPNetwork_Critic(num_in_critic, 1,
                                                    hidden_dim=hidden_dim,
                                                    norm_in= self.norm_in,agent=self,n_atoms=n_atoms,D4PG=D4PG,TD3=TD3,maddpg=maddpg)
            if torch.cuda.device_count() > 1 and maddpg.data_parallel:
                self.critic = nn.DataParallel(self.critic)
                self.target_critic = nn.DataParallel(self.target_critic)


        if not maddpg.only_policy:
            hard_update(self.target_policy, self.policy)
            hard_update(self.target_critic, self.critic)

        self.a_lr = a_lr
        self.c_lr = c_lr
        self.EM_lr = EM_lr
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=a_lr, weight_decay =0)
  
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=c_lr, weight_decay=0)
        if not maddpg.only_policy:
            self.critic_optimizer = Adam(self.critic.parameters(), lr=c_lr, weight_decay=0)
        if I2A:
            self.EM_optimizer = Adam(self.EM.parameters(), lr=EM_lr)
            self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=a_lr, weight_decay =0)
            self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=a_lr, weight_decay =0)
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

    def change_a_lr(self,lr):
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = lr
        #self.policy_optimizer = Adam(self.policy.parameters(), lr=lr, weight_decay =0)
    def change_c_lr(self,lr):
        #self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, weight_decay =0)
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = lr
    def step(self, obs,ran,acs=None,explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
    
        # mixed disc/cont
        if explore:
            if not ran: # not random
                if self.I2A:
                    action = acs
                else:
                    action = self.policy(obs)
                if self.counter % 1000 == 0:
                    print(torch.softmax(action[:,:self.action_dim],dim=1))
                a = gumbel_softmax(action[:,:self.action_dim].view(1,self.action_dim),hard=True, device=self.maddpg.torch_device)
                p = torch.clamp((action[:,self.action_dim:].view(1,self.param_dim) + Variable(processor(Tensor(self.exploration.noise()),device=self.device,torch_device=self.maddpg.torch_device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
                action = torch.cat((a,p),1) 
                self.counter +=1

            else: # random
                action = torch.cat((onehot_from_logits(torch.empty((1,self.action_dim),device=self.maddpg.torch_device,requires_grad=False).uniform_(-1,1)),
                            torch.empty((1,self.param_dim),device=self.maddpg.torch_device,requires_grad=False).uniform_(-1,1) ),1)
        else:
            if self.I2A:
                action = acs
            else:
                action = self.policy(obs)
            if self.counter % 1000 == 0:
                print(torch.softmax(action[:,:self.action_dim],dim=1))
            a = onehot_from_logits(action[0,:self.action_dim].view(1,self.action_dim))
            #p = torch.clamp(action[0,self.action_dim:].view(1,self.param_dim),min=-1.0,max=1.0) # get noisey params (OU)
            p = action[0,self.action_dim:].view(1,self.param_dim)
            #p = torch.clamp((action[0,self.action_dim:].view(1,self.param_dim) + Variable(processor(Tensor(self.exploration.noise()),device=self.device,torch_device=self.maddpg.torch_device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
            action = torch.cat((a,p),1) 
            self.counter +=1

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
        #self.maddpg.prep_training(device='cpu')  # move parameters to CPU before saving
        if self.maddpg.data_parallel:

            dict =  {'policy': self.policy.module.state_dict(),
                    'critic': self.critic.module.state_dict(),
                    'target_policy': self.target_policy.module.state_dict(),
                    'target_critic': self.target_critic.module.state_dict(),


                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),}
        else:
            
            dict =  {'policy': self.policy.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_policy': self.target_policy.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),}
                    

        if self.I2A:
            dict = {'policy': self.policy.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_policy': self.target_policy.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    'EM':self.EM.state_dict(),
                    'EM_optimizer':self.EM_optimizer.state_dict(),
                    'policy_prime_optimizer': self.policy_prime_optimizer.state_dict(),
                    'imagination_policy_optimizer': self.imagination_policy_optimizer.state_dict(),
                    'policy_prime':  self.policy_prime.state_dict(),
                    'imagination_policy': self.imagination_policy.state_dict()}

        return dict

    #Used to get just the actor weights and params.
    def get_actor_params(self):
        if self.maddpg.data_parallel:
            return {'policy': self.policy.module.state_dict()}
        else:
            return {'policy': self.policy.state_dict()}


    #Used to get just the critic weights and params.
    def get_critic_params(self):
        if self.maddpg.data_parallel:
            return {'critic': self.critic.module.state_dict()}
        else:
            return {'critic': self.critic.state_dict()}

        

    def load_params(self, params):
        if self.device == 'cuda':
            dev = self.maddpg.torch_device
        else:
            dev = torch.device('cpu')
        if self.maddpg.data_parallel:

            self.policy.module.load_state_dict(params['policy'])
            self.critic.module.load_state_dict(params['critic'])
            self.target_policy.module.load_state_dict(params['target_policy'])
            self.target_critic.module.load_state_dict(params['target_critic'])
        else:

            self.policy.load_state_dict(params['policy'])
            self.critic.load_state_dict(params['critic'])
            self.target_policy.load_state_dict(params['target_policy'])
            self.target_critic.load_state_dict(params['target_critic'])
            

        if self.I2A:
            self.EM.load_state_dict(params['EM'])
            self.policy_prime.load_state_dict(params['policy_prime'])
            self.imagination_policy.load_state_dict(params['imagination_policy'])

        self.policy.to(dev)
        self.critic.to(dev)
        self.target_policy.to(dev)
        self.target_critic.to(dev)

        if self.I2A:
            self.EM.to(dev)
            self.policy_prime.to(dev)
            self.imagination_policy.to(dev)
        
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)
        if self.I2A:
            self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=self.a_lr, weight_decay =0)
            self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=self.a_lr, weight_decay =0)
            self.EM_optimizer = Adam(self.EM.parameters(), lr=self.EM_lr)

        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

        if self.I2A:
            self.EM_optimizer.load_state_dict(params['EM_optimizer'])
            self.policy_prime_optimizer.load_state_dict(params['policy_prime_optimizer'])
            self.imagination_policy_optimizer.load_state_dict(params['imagination_policy_optimizer'])
        
        
    def load_target_policy_params(self, params):
        if self.device == 'cuda':
            dev = torch.device(self.maddpg.torch_device)
        else:
            dev = torch.device('cpu')
        if self.maddpg.data_parallel:
            self.target_policy.module.load_state_dict(params['target_policy'])
            self.policy.module.load_state_dict(params['policy'])

        else:
            self.target_policy.load_state_dict(params['target_policy'])
            self.policy.load_state_dict(params['policy'])



        if self.I2A:
            self.EM.load_state_dict(params['EM'])
            self.policy_prime.load_state_dict(params['policy_prime'])
            self.imagination_policy.load_state_dict(params['imagination_policy'])


        self.policy.to(dev)
        if self.I2A:
            self.EM.to(dev)
            self.imagination_policy.to(dev)
            self.policy_prime.to(dev)

           
    def load_policy_params(self, params):
        if self.device == 'cuda':
            dev = torch.device(self.maddpg.torch_device)
        else:
            dev = torch.device('cpu')
        if self.maddpg.data_parallel:
            self.policy.module.load_state_dict(params['policy'])

        else:
            self.policy.load_state_dict(params['policy'])



        if self.I2A:
            self.EM.load_state_dict(params['EM'])
            self.policy_prime.load_state_dict(params['policy_prime'])
            self.imagination_policy.load_state_dict(params['imagination_policy'])


        self.policy.to(dev)
        if self.I2A:
            self.EM.to(dev)
            self.imagination_policy.to(dev)
            self.policy_prime.to(dev)

         
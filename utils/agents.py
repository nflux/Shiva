from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch
from .networks import MLPNetwork_Actor,MLPNetwork_Critic,I2A_Network,LSTMNetwork_Critic,Dimension_Reducer,LSTM_Actor
import torch.nn.functional as F
import torch.nn as nn
from .i2a import *
from .misc import hard_update, gumbel_softmax, onehot_from_logits,processor,e_greedy_bool,zero_params
from .noise import OUNoise
import numpy as np

def init_agents(num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
            num_in_reducer, config=None, maddpg=object, only_policy=False):
    
    if config.lstm_pol and config.lstm_crit:
        return RecAgent(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, 
                        num_out_EM, num_in_reducer, maddpg)
    elif config.lstm_pol:
        return RecPolAgent(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, 
                        num_out_EM, num_in_reducer, maddpg)
    elif config.lstm_crit:
        return RecCritAgent(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, 
                        num_out_EM, num_in_reducer, maddpg)
    else:
        return NonRecAgent(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, 
                        num_out_EM, num_in_reducer, maddpg)

class Base_Agent(object):
    def __init__(self, config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                num_in_reducer, maddpg=object, only_policy=False):
        self.config = config
        self.num_in_pol = num_in_pol
        self.num_out_pol = num_out_pol
        self.num_in_critic = num_in_critic
        self.num_in_EM = num_in_EM
        self.num_out_EM = num_out_EM
        self.num_in_reducer = num_in_reducer
        self.maddpg = maddpg
        self.I2A = config.i2a
        self.norm_in = False
        self.counter = 0
        self.updating_actor = False
        self.maddpg = maddpg
        self.batch_size = config.batch_size
        self.param_dim = config.param_dim
        self.action_dim = config.ac_dim
        self.imagination_policy_branch = config.imag_pol_branch
        self.device = config.device
        self.n_branches = 1 + self.imagination_policy_branch # number of imagination branches
        self.delta = config.delta_z
        # D4PG
        self.n_atoms = config.n_atoms
        self.vmax = config.vmax
        self.vmin = config.vmin
        self.world_status_dim = config.world_status_dim
        self.LSTM_crit = config.lstm_crit
        self.LSTM_policy = config.lstm_pol
        self.a_lr = config.a_lr
        self.c_lr = config.c_lr
        self.EM_lr = config.em_lr
        self.rollout_steps = config.roll_steps

        self.I2A_num_in_pol = num_in_pol
        self.hidden_dim_lstm = config.hidden_dim_lstm
        self.num_total_out_EM = num_out_EM + self.world_status_dim + 1
        if self.I2A:
            self.EM = EnvironmentModel(num_in_EM, num_out_EM,hidden_dim=config.hidden_dim,
                                  norm_in=self.norm_in,agent=self,maddpg=maddpg)

            self.imagination_policy = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                    hidden_dim=config.hidden_dim,
                                    discrete_action=config.discrete_action, 
                                    norm_in= self.norm_in,agent=self,maddpg=maddpg)
        # policy prime for I2A
            self.policy_prime = MLPNetwork_Actor(num_in_pol, num_out_pol,
                                    hidden_dim=config.hidden_dim,
                                    discrete_action=config.discrete_action, 
                                    norm_in= self.norm_in,agent=self,maddpg=maddpg)
            
            self.EM_optimizer = Adam(self.EM.parameters(), lr=self.EM_lr)
            self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=self.a_lr, weight_decay =0)
            self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=self.a_lr, weight_decay =0)
        else:
            self.EM = None
            self.imagination_policy = None
            self.policy_prime = None
        
        # self.reducer = Dimension_Reducer(num_in_reducer,agent=self,maddpg=maddpg,norm_in=self.norm_in)
        # self.reducer_optimizer = Adam(self.reducer.parameters(), lr=self.a_lr, weight_decay =0)
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy = None
        self.critic = None
        self.target_policy = None
        self.target_critic = None
        self.policy_optimizer = None
        self.critic_optimizer = None

        if not config.discrete_action: # input to OUNoise is size of param space # TODO change OUNoise param to # params
            self.exploration = OUNoise(self.param_dim) # hard coded
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = config.discrete_action
    
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
    
    def get_params(self):
        #self.maddpg.prep_training(device='cpu')  # move parameters to CPU before saving
        if self.config.data_parallel:

            dict =  {'policy': self.policy.module.state_dict(),
                    'critic': self.critic.module.state_dict(),
                    'target_policy': self.target_policy.module.state_dict(),
                    'target_critic': self.target_critic.module.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    #'reducer': self.reducer.module.state_dict(),
                    #'reducer_optimizer': self.reducer_optimizer.state_dict(),
                    }
        else:
            
            dict =  {'policy': self.policy.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_policy': self.target_policy.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'policy_optimizer': self.policy_optimizer.state_dict(),
                    'critic_optimizer': self.critic_optimizer.state_dict(),
                    #'reducer':self.reducer.state_dict(),
                    #'reducer_optimizer': self.reducer_optimizer.state_dict()}
                    }
                    

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
                    'imagination_policy': self.imagination_policy.state_dict(),
                   #'reducer': self.reducer.state_dict(),
                   #'reducer_optimizer': self.reducer_optimizer.state_dict()}
                    }

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
            #self.reducer.module.load_state_dict(params['reducer'])

            self.critic.module.load_state_dict(params['critic'])
            self.target_policy.module.load_state_dict(params['target_policy'])
            self.target_critic.module.load_state_dict(params['target_critic'])
        else:

            self.policy.load_state_dict(params['policy'])
            #self.reducer.load_state_dict(params['reducer'])

            self.critic.load_state_dict(params['critic'])
            self.target_policy.load_state_dict(params['target_policy'])
            self.target_critic.load_state_dict(params['target_critic'])
            

        if self.I2A:
            self.EM.load_state_dict(params['EM'])
            self.policy_prime.load_state_dict(params['policy_prime'])
            self.imagination_policy.load_state_dict(params['imagination_policy'])

        self.policy.to(dev)
        #self.reducer.to(dev)
        self.critic.to(dev)
        self.target_policy.to(dev)
        self.target_critic.to(dev)

        if self.I2A:
            self.EM.to(dev)
            self.policy_prime.to(dev)
            self.imagination_policy.to(dev)
        
        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)
        #self.reducer_optimizer = Adam(self.reducer.parameters(),lr=self.a_lr,weight_decay = 0)
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
        if self.config.data_parallel:
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
        if self.config.data_parallel:
            self.policy.module.load_state_dict(params['policy'])
            #self.reducer.module.load_state_dict(params['reducer'])

        else:
            self.policy.load_state_dict(params['policy'])
            #self.reducer.load_state_dict(params['reducer'])



        if self.I2A:
            self.EM.load_state_dict(params['EM'])
            self.policy_prime.load_state_dict(params['policy_prime'])
            self.imagination_policy.load_state_dict(params['imagination_policy'])


        self.policy.to(dev)
        if self.I2A:
            self.EM.to(dev)
            self.imagination_policy.to(dev)
            self.policy_prime.to(dev)

    def step(self, obs,ran,acs = None,explore=False):
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
                    print(torch.softmax(action.view(-1)[:self.action_dim],dim=0))
                
                if self.config.lstm_pol:
                    a = gumbel_softmax(torch.log_softmax(action[:,:,:self.action_dim],dim=2),hard=True, device=self.maddpg.torch_device,LSTM=True).view(1,self.action_dim)
                    p = torch.clamp((torch.squeeze(action[:,:,self.action_dim:]).view(1,self.param_dim) + Variable(processor(Tensor(self.exploration.noise()),device=self.device,torch_device=self.maddpg.torch_device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
                else:
                    a = gumbel_softmax(torch.log_softmax(action[:,:self.action_dim],dim=2).view(1,self.action_dim),hard=True, device=self.maddpg.torch_device)
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
            if self.counter % 5000 == 0:
                print(torch.softmax(action.view(-1)[:self.action_dim],dim=0))
            if self.config.lstm_pol:
                a = onehot_from_logits(action[:,:,:self.action_dim],LSTM=True).view(1,self.action_dim)
                #p = torch.clamp(action[0,self.action_dim:].view(1,self.param_dim),min=-1.0,max=1.0) # get noisey params (OU)
                p = torch.squeeze(action[:,:,self.action_dim:]).view(1,self.param_dim)

            else:
                a = onehot_from_logits(action[:,:self.action_dim].view(1,self.action_dim))
                #p = torch.clamp(action[0,self.action_dim:].view(1,self.param_dim),min=-1.0,max=1.0) # get noisey params (OU)
                p = action[:,self.action_dim:].view(1,self.param_dim)
                #p = torch.clamp((action[0,self.action_dim:].view(1,self.param_dim) + Variable(processor(Tensor(self.exploration.noise()),device=self.device,torch_device=self.maddpg.torch_device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
            action = torch.cat((a,p),1) 
            self.counter +=1

        #if self.maddpg.zero_critic:
        #    action = zero_params(action)
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
        
class RecPolAgent(Base_Agent):
    def __init__(self, config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                num_in_reducer, maddpg=object, only_policy=False):
        super().__init__(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                        num_in_reducer, maddpg, only_policy)
        self.policy = LSTM_Actor(config,self.I2A_num_in_pol, num_out_pol, self.num_total_out_EM,
                                hidden_dim=config.hidden_dim,
                                discrete_action=config.discrete_action,
                                norm_in= self.norm_in,agent=self,I2A=self.I2A,rollout_steps=rollout_steps,
                                EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                LSTM_hidden=config.lstm_hidden,maddpg=maddpg)

        if torch.cuda.device_count() > 1 and maddpg.data_parallel:
            self.policy = nn.DataParallel(self.policy)
        
        if not only_policy:
            self.target_policy = LSTM_Actor(config,self.I2A_num_in_pol, num_out_pol,self.num_total_out_EM,
                                                    hidden_dim=config.hidden_dim,discrete_action=config.discrete_action,
                                                    norm_in= self.norm_in,agent=self,I2A=self.I2A,
                                                    rollout_steps=rollout_steps,EM = self.EM, 
                                                    pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                                    LSTM_hidden=config.lstm_hidden,maddpg=maddpg)
        
            if torch.cuda.device_count() > 1 and maddpg.data_parallel:
                self.target_policy = nn.DataParallel(self.target_policy)
        
            self.critic = MLPNetwork_Critic(config, num_in_critic, 1,
                                        hidden_dim=config.hidden_dim,
                                        norm_in= self.norm_in,agent=self,n_atoms=config.n_atoms,
                                        D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)

            self.target_critic = MLPNetwork_Critic(config, num_in_critic, 1,
                                                    hidden_dim=hidden_dim,
                                                    norm_in= self.norm_in,agent=self,n_atoms=config.n_atoms,
                                                    D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)

            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.critic = nn.DataParallel(self.critic)
                self.target_critic = nn.DataParallel(self.target_critic)
            
            hard_update(self.target_policy, self.policy)
            hard_update(self.target_critic, self.critic)

            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)

        self.critic_grad_by_action = np.zeros(self.param_dim)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)
            

        
class RecCritAgent(Base_Agent):
    def __init__(self, config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                num_in_reducer, maddpg=object, only_policy=False):
        super().__init__(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                        num_in_reducer, maddpg, only_policy)
        
        self.policy = I2A_Network(config,self.I2A_num_in_pol, num_out_pol, self.num_total_out_EM,
                        hidden_dim=config.hidden_dim,
                        discrete_action=config.discrete_action,
                        norm_in= self.norm_in,agent=self,I2A=self.I2A,rollout_steps=self.rollout_steps,
                        EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                        LSTM_hidden=config.lstm_hidden,maddpg=maddpg)
        
        if torch.cuda.device_count() > 1 and config.data_parallel:
            self.policy = nn.DataParallel(self.policy)
        
        if not only_policy:
            self.target_policy = I2A_Network(config,self.I2A_num_in_pol, num_out_pol,self.num_total_out_EM,
                                                hidden_dim=config.hidden_dim,
                                                discrete_action=config.discrete_action,
                                                norm_in= self.norm_in,agent=self,I2A=self.I2A,rollout_steps=self.rollout_steps,
                                                EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                                LSTM_hidden=config.lstm_hidden,maddpg=maddpg)
            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.target_policy = nn.DataParallel(self.target_policy)


            self.critic = LSTMNetwork_Critic(config, num_in_critic, 1,
                                            hidden_dim=config.hidden_dim, agent=self, n_atoms=config.n_atoms, 
                                            D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)
            self.target_critic = LSTMNetwork_Critic(config, num_in_critic, 1,
                                            hidden_dim=config.hidden_dim, agent=self, n_atoms=config.n_atoms, 
                                            D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)
            
            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.critic = nn.DataParallel(self.critic)
                self.target_critic = nn.DataParallel(self.target_critic)
            
            hard_update(self.target_policy, self.policy)
            hard_update(self.target_critic, self.critic)

            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=a_lr, weight_decay =0)



class RecAgent(Base_Agent):
    def __init__(self, config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                num_in_reducer, maddpg=object, only_policy=False):
        super().__init__(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                        num_in_reducer, maddpg, only_policy)
        
        self.policy = LSTM_Actor(config,self.I2A_num_in_pol, num_out_pol, self.num_total_out_EM,
                                hidden_dim=config.hidden_dim,
                                discrete_action=config.discrete_action,
                                norm_in= self.norm_in,agent=self,I2A=self.I2A,rollout_steps=self.rollout_steps,
                                EM = self.EM, pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                LSTM_hidden=config.lstm_hidden,maddpg=maddpg)

        if torch.cuda.device_count() > 1 and config.data_parallel:
            self.policy = nn.DataParallel(self.policy)
        
        if not only_policy:
            self.target_policy = LSTM_Actor(config,self.I2A_num_in_pol, num_out_pol,self.num_total_out_EM,
                                            hidden_dim=config.hidden_dim,discrete_action=config.discrete_action,
                                            norm_in= self.norm_in,agent=self,I2A=self.I2A,
                                            rollout_steps=self.rollout_steps,EM = self.EM, 
                                            pol_prime = self.policy_prime,imagined_pol = self.imagination_policy,
                                            LSTM_hidden=config.lstm_hidden,maddpg=maddpg)

            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.target_policy = nn.DataParallel(self.target_policy)
            
            self.critic = LSTMNetwork_Critic(config, num_in_critic, 1,
                                            hidden_dim=config.hidden_dim, agent=self, n_atoms=config.n_atoms, 
                                            D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)
            self.target_critic = LSTMNetwork_Critic(config, num_in_critic, 1,
                                            hidden_dim=config.hidden_dim, agent=self, n_atoms=config.n_atoms, 
                                            D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)
            
            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.critic = nn.DataParallel(self.critic)
                self.target_critic = nn.DataParallel(self.target_critic)
            
            hard_update(self.target_policy, self.policy)
            hard_update(self.target_critic, self.critic)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)                    

class NonRecAgent(Base_Agent):
    def __init__(self, config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                num_in_reducer, maddpg=object, only_policy=False):
        super().__init__(config, num_in_pol, num_out_pol, num_in_critic, num_in_EM, num_out_EM, 
                        num_in_reducer, maddpg, only_policy)
        
        self.policy = I2A_Network(config,self.I2A_num_in_pol, num_out_pol, self.num_total_out_EM,
                        hidden_dim=config.hidden_dim,
                        discrete_action=config.discrete_action,
                        norm_in= self.norm_in,agent=self,I2A=self.I2A,
                        rollout_steps=self.rollout_steps,
                        EM = self.EM, pol_prime = self.policy_prime,
                        imagined_pol = self.imagination_policy,
                        LSTM_hidden=config.lstm_hidden,maddpg=maddpg)

        if torch.cuda.device_count() > 1 and config.data_parallel:
            self.policy = nn.DataParallel(self.policy)
        
        if not only_policy:
            self.target_policy = I2A_Network(config,self.I2A_num_in_pol, num_out_pol,self.num_total_out_EM,
                                            hidden_dim=config.hidden_dim,
                                            discrete_action=config.discrete_action,
                                            norm_in= self.norm_in,agent=self,I2A=self.I2A,
                                            rollout_steps=self.rollout_steps,
                                            EM = self.EM, pol_prime = self.policy_prime,
                                            imagined_pol = self.imagination_policy,
                                            LSTM_hidden=config.lstm_hidden,maddpg=maddpg)
            
            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.target_policy = nn.DataParallel(self.target_policy)
            
            self.critic = MLPNetwork_Critic(config, num_in_critic, 1,
                                        hidden_dim=config.hidden_dim,
                                        norm_in= self.norm_in,agent=self,
                                        n_atoms=config.n_atoms,D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)

            self.target_critic = MLPNetwork_Critic(config, num_in_critic, 1,
                                        hidden_dim=config.hidden_dim,
                                        norm_in= self.norm_in,agent=self,
                                        n_atoms=config.n_atoms,D4PG=config.d4pg,TD3=config.td3,maddpg=maddpg)

            if torch.cuda.device_count() > 1 and config.data_parallel:
                self.critic = nn.DataParallel(self.critic)
                self.target_critic = nn.DataParallel(self.target_critic)
            
            hard_update(self.target_policy, self.policy)
            hard_update(self.target_critic, self.critic)

            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)
        
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)

#     def get_params(self):
#         #self.maddpg.prep_training(device='cpu')  # move parameters to CPU before saving
#         if self.config.data_parallel:

#             dict =  {'policy': self.policy.module.state_dict(),
#                     'critic': self.critic.module.state_dict(),
#                     'target_policy': self.target_policy.module.state_dict(),
#                     'target_critic': self.target_critic.module.state_dict(),
#                     'policy_optimizer': self.policy_optimizer.state_dict(),
#                     'critic_optimizer': self.critic_optimizer.state_dict(),
#                     #'reducer': self.reducer.module.state_dict(),
#                     #'reducer_optimizer': self.reducer_optimizer.state_dict(),
# }
#         else:
            
#             dict =  {'policy': self.policy.state_dict(),
#                     'critic': self.critic.state_dict(),
#                     'target_policy': self.target_policy.state_dict(),
#                     'target_critic': self.target_critic.state_dict(),
#                     'policy_optimizer': self.policy_optimizer.state_dict(),
#                     'critic_optimizer': self.critic_optimizer.state_dict(),
#                     #'reducer':self.reducer.state_dict(),
#                     #'reducer_optimizer': self.reducer_optimizer.state_dict()}
#                     }
                    

#         if self.I2A:
#             dict = {'policy': self.policy.state_dict(),
#                     'critic': self.critic.state_dict(),
#                     'target_policy': self.target_policy.state_dict(),
#                     'target_critic': self.target_critic.state_dict(),
#                     'policy_optimizer': self.policy_optimizer.state_dict(),
#                     'critic_optimizer': self.critic_optimizer.state_dict(),
#                     'EM':self.EM.state_dict(),
#                     'EM_optimizer':self.EM_optimizer.state_dict(),
#                     'policy_prime_optimizer': self.policy_prime_optimizer.state_dict(),
#                     'imagination_policy_optimizer': self.imagination_policy_optimizer.state_dict(),
#                     'policy_prime':  self.policy_prime.state_dict(),
#                     'imagination_policy': self.imagination_policy.state_dict(),
#                    #'reducer': self.reducer.state_dict(),
#                    #'reducer_optimizer': self.reducer_optimizer.state_dict()}
#                     }

#         return dict

#     #Used to get just the actor weights and params.
#     def get_actor_params(self):
#         if self.maddpg.data_parallel:
#             return {'policy': self.policy.module.state_dict()}
#         else:
#             return {'policy': self.policy.state_dict()}


#     #Used to get just the critic weights and params.
#     def get_critic_params(self):
#         if self.maddpg.data_parallel:
#             return {'critic': self.critic.module.state_dict()}
#         else:
#             return {'critic': self.critic.state_dict()}

        

#     def load_params(self, params):
#         if self.device == 'cuda':
#             dev = self.maddpg.torch_device
#         else:
#             dev = torch.device('cpu')
#         if self.maddpg.data_parallel:

#             self.policy.module.load_state_dict(params['policy'])
#             #self.reducer.module.load_state_dict(params['reducer'])

#             self.critic.module.load_state_dict(params['critic'])
#             self.target_policy.module.load_state_dict(params['target_policy'])
#             self.target_critic.module.load_state_dict(params['target_critic'])
#         else:

#             self.policy.load_state_dict(params['policy'])
#             #self.reducer.load_state_dict(params['reducer'])

#             self.critic.load_state_dict(params['critic'])
#             self.target_policy.load_state_dict(params['target_policy'])
#             self.target_critic.load_state_dict(params['target_critic'])
            

#         if self.I2A:
#             self.EM.load_state_dict(params['EM'])
#             self.policy_prime.load_state_dict(params['policy_prime'])
#             self.imagination_policy.load_state_dict(params['imagination_policy'])

#         self.policy.to(dev)
#         #self.reducer.to(dev)
#         self.critic.to(dev)
#         self.target_policy.to(dev)
#         self.target_critic.to(dev)

#         if self.I2A:
#             self.EM.to(dev)
#             self.policy_prime.to(dev)
#             self.imagination_policy.to(dev)
        
#         self.critic_grad_by_action = np.zeros(self.param_dim)
#         self.policy_optimizer = Adam(self.policy.parameters(), lr=self.a_lr, weight_decay =0)
#         #self.reducer_optimizer = Adam(self.reducer.parameters(),lr=self.a_lr,weight_decay = 0)
#         self.critic_optimizer = Adam(self.critic.parameters(), lr=self.c_lr, weight_decay=0)
#         if self.I2A:
#             self.policy_prime_optimizer = Adam(self.policy_prime.parameters(), lr=self.a_lr, weight_decay =0)
#             self.imagination_policy_optimizer = Adam(self.imagination_policy.parameters(), lr=self.a_lr, weight_decay =0)
#             self.EM_optimizer = Adam(self.EM.parameters(), lr=self.EM_lr)

#         self.policy_optimizer.load_state_dict(params['policy_optimizer'])
#         self.critic_optimizer.load_state_dict(params['critic_optimizer'])

#         if self.I2A:
#             self.EM_optimizer.load_state_dict(params['EM_optimizer'])
#             self.policy_prime_optimizer.load_state_dict(params['policy_prime_optimizer'])
#             self.imagination_policy_optimizer.load_state_dict(params['imagination_policy_optimizer'])
        
        
#     def load_target_policy_params(self, params):
#         if self.device == 'cuda':
#             dev = torch.device(self.maddpg.torch_device)
#         else:
#             dev = torch.device('cpu')
#         if self.maddpg.data_parallel:
#             self.target_policy.module.load_state_dict(params['target_policy'])
#             self.policy.module.load_state_dict(params['policy'])

#         else:
#             self.target_policy.load_state_dict(params['target_policy'])
#             self.policy.load_state_dict(params['policy'])



#         if self.I2A:
#             self.EM.load_state_dict(params['EM'])
#             self.policy_prime.load_state_dict(params['policy_prime'])
#             self.imagination_policy.load_state_dict(params['imagination_policy'])


#         self.policy.to(dev)
#         if self.I2A:
#             self.EM.to(dev)
#             self.imagination_policy.to(dev)
#             self.policy_prime.to(dev)

           
#     def load_policy_params(self, params):
#         if self.device == 'cuda':
#             dev = torch.device(self.maddpg.torch_device)
#         else:
#             dev = torch.device('cpu')
#         if self.maddpg.data_parallel:
#             self.policy.module.load_state_dict(params['policy'])
#             #self.reducer.module.load_state_dict(params['reducer'])

#         else:
#             self.policy.load_state_dict(params['policy'])
#             #self.reducer.load_state_dict(params['reducer'])



#         if self.I2A:
#             self.EM.load_state_dict(params['EM'])
#             self.policy_prime.load_state_dict(params['policy_prime'])
#             self.imagination_policy.load_state_dict(params['imagination_policy'])


#         self.policy.to(dev)
#         if self.I2A:
#             self.EM.to(dev)
#             self.imagination_policy.to(dev)
#             self.policy_prime.to(dev)


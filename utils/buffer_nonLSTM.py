import numpy as np
from torch import Tensor,squeeze
from torch.autograd import Variable
import operator
import torch
import math
from utils.buffer import ReplayBuffer

class ReplayBufferNonLSTM(ReplayBuffer):
    """
    Non-LSTM Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dim, ac_dim, batch_size, seq_length,overlap,hidden_dim_lstm,k,prox_item_size,SIL,pretrain=False):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        #This initializes the parent buffer methods/variables
        super().__init__( max_steps, num_agents, obs_dim, ac_dim, hidden_dim_lstm,k,prox_item_size,SIL,pretrain)
        
        # self.max_episodes = int(max_steps/episode_length)
        self.start_loc = 0
        self.obs_acs_dim = obs_dim + ac_dim
        self.prox_item_size_per_agent = prox_item_size//(2*num_agents)

        self.obs_buffs = torch.zeros((max_steps, num_agents, obs_dim))
        self.ac_buffs = torch.zeros((max_steps, num_agents, ac_dim),requires_grad=False)
        self.n_step_buffs = torch.zeros((max_steps, num_agents, 1),requires_grad=False)
        self.rew_buffs = torch.zeros((max_steps, num_agents, 1),requires_grad=False)
        self.mc_buffs = torch.zeros((max_steps, num_agents, 1),requires_grad=False)
        self.next_obs_buffs = torch.zeros((max_steps, num_agents, obs_dim),requires_grad=False)
        self.done_buffs = torch.zeros((max_steps, num_agents, 1),requires_grad=False)
        self.ws_buffs = torch.zeros((max_steps, num_agents, 1),requires_grad=False)
        self.SIL_priorities = torch.zeros((max_steps,num_agents,1),requires_grad=False)
        self.ensemble_priorities = torch.zeros((max_steps,num_agents,k),requires_grad=False)
    
    def push(self, exps):
        nentries = len(exps)
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs_buffs = roll(self.obs_buffs, rollover)
            self.ac_buffs = roll(self.ac_buffs, rollover)
            self.rew_buffs = roll(self.rew_buffs, rollover)
            self.next_obs_buffs = roll(self.next_obs_buffs, rollover)
            self.done_buffs = roll(self.done_buffs, rollover)
            self.mc_buffs = roll(self.mc_buffs, rollover)
            self.n_step_buffs = roll(self.n_step_buffs, rollover)
            self.ws_buffs = roll(self.ws_buffs, rollover)
            self.ensemble_priorities = roll(self.ensemble_priorities,rollover)
            self.SIL_priorities = roll(self.SIL_priorities,rollover)


            self.curr_i = 0
            self.filled_i = self.max_steps

        # Used just for readability
        oa_i = self.obs_dim + self.ac_dim
        rew_i = oa_i+1
        next_oi = rew_i+self.obs_dim

        self.obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, :self.obs_dim]
        self.ac_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.ac_dim] = exps[:, :, self.obs_dim:oa_i]
        self.rew_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, oa_i:rew_i]
        self.next_obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, rew_i:next_oi]
        self.done_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi:next_oi+1]
        self.mc_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+1:next_oi+2]
        self.n_step_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+2:next_oi+3]
        self.ws_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, next_oi+3:next_oi+4]
        self.ensemble_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = exps[:, :, next_oi+4:next_oi+4+self.k]
        self.SIL_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = exps[:, :, next_oi+4+self.k:next_oi+4+self.k+1]

        
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0
       
    
    def merge_buffer(self, buffer):
        nentries = len(buffer)
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.obs_buffs = roll(self.obs_buffs, rollover)
            self.ac_buffs = roll(self.ac_buffs, rollover)
            self.rew_buffs = roll(self.rew_buffs, rollover)
            self.next_obs_buffs = roll(self.next_obs_buffs, rollover)
            self.done_buffs = roll(self.done_buffs, rollover)
            self.mc_buffs = roll(self.mc_buffs, rollover)
            self.n_step_buffs = roll(self.n_step_buffs, rollover)
            self.ws_buffs = roll(self.ws_buffs, rollover)
            self.ensemble_priorities = roll(self.ensemble_priorities,rollover)
            self.SIL_priorities = roll(self.SIL_priorities,rollover)


            self.curr_i = 0
            self.filled_i = self.max_steps

        # Used just for readability
        oa_i = self.obs_dim + self.ac_dim
        rew_i = oa_i+1
        next_oi = rew_i+self.obs_dim

        self.obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = buffer.obs_buffs[:nentries, :self.num_agents, :self.obs_dim]
        self.ac_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.ac_dim] = buffer.ac_buffs[:nentries, :self.num_agents, :self.ac_dim]
        self.rew_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.rew_buffs[:nentries, :self.num_agents, :1]
        self.next_obs_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.obs_dim] = buffer.next_obs_buffs[:nentries, :self.num_agents, :self.obs_dim]
        self.done_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.done_buffs[:nentries, :self.num_agents, :1]
        self.mc_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.mc_buffs[:nentries, :self.num_agents, :1]
        self.n_step_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] =buffer.n_step_buffs[:nentries, :self.num_agents, :1]
        self.ws_buffs[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = buffer.ws_buffs[:nentries, :self.num_agents, :1]
        self.ensemble_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = buffer.ensemble_priorities[:nentries, :self.num_agents, : self.k]
        self.SIL_priorities[self.curr_i:self.curr_i+nentries, :self.num_agents, :self.k] = buffer.SIL_priorities[:nentries, :self.num_agents,  :self.k]

        
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self,inds, to_gpu=False, norm_rews=False,device="cuda:0"):

        if to_gpu:
            cast = lambda x: Variable(x,requires_grad=False).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(x,requires_grad=True).to(device) # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(x, requires_grad=True)


        if norm_rews:
            ret_rews = [cast((self.rew_buffs[inds, i, :] -
                             self.rew_buffs[:self.filled_i, i, :].mean()) /
                             self.rew_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]

            ret_mc = [cast((self.mc_buffs[inds, i, :] -
                              self.mc_buffs[:self.filled_i, i, :].mean()) /
                             self.mc_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]

            ret_n_step = [cast((self.n_step_buffs[inds, i, :] -
                              self.n_step_buffs[:self.filled_i, i, :].mean()) /
                             self.n_step_buffs[:self.filled_i, i, :].std())
                        for i in range(self.num_agents)]
            
        else:
            ret_rews = [cast(self.rew_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]
            ret_mc = [cast(self.mc_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]
            ret_n_step = [cast(self.n_step_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)]


        return ([cast_obs(self.obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[inds, i, :]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                [cast(self.done_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                ret_mc,
                ret_n_step,
                [cast(self.ws_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)])

import numpy as np
from torch import Tensor,squeeze
from torch.autograd import Variable
import operator
import torch
import math
from utils.buffer import ReplayBuffer

class ReplayBufferLSTM(ReplayBuffer):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
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
        
        self.seq_length = seq_length
        self.overlap = overlap
        
        self.max_steps = int(max_steps) # Max sequences did this to reduce var name change
        self.seq_exps = torch.zeros((seq_length, self.max_steps, num_agents, self.total_dim), requires_grad=False)

    def merge_buffer(self, buffer):
        nentries = len(buffer)

        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.seq_exps = roll2(self.seq_exps, rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps
        
        self.seq_exps[:self.seq_length, self.curr_i:self.curr_i+nentries, :self.num_agents, :] = buffer.seq_exps[:self.seq_length, :nentries, :self.num_agents, :]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def push(self, exps):
        ep_length = len(exps)
        nentries = (math.ceil(ep_length/self.overlap)) - 1 # Number of rows to input in seq_exps tensor
    
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            self.seq_exps = roll2(self.seq_exps, rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps

        # NOTE: Exps include: obs, acs, n_step_rew, n_step_done, MC_targets, n_step_targets, 
        #       n_step_ws, ensemble priorities, replay priorities
        if ep_length % self.overlap == 0:
            for n in range(nentries):
                start_pt = n*self.overlap
                self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[start_pt:start_pt+self.seq_length, :, :]
        else:
            for n in range(nentries):
                if n != nentries-1:
                    start_pt = n*self.overlap
                    self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[start_pt:start_pt+self.seq_length, :, :]
                else:
                    # Get the last values if the episode length is not evenly divisible by the overlap amount
                    self.seq_exps[:self.seq_length, self.curr_i+n, :self.num_agents, :] = exps[ep_length-self.seq_length:, :, :]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, inds, to_gpu=False, device="cuda:0"):

        if to_gpu:
            cast = lambda x: Variable(x, requires_grad=False).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
        if to_gpu:
            cast_obs = lambda x: Variable(x, requires_grad=True).to(device) # obs need gradient for cent-Q
        else:
            cast_obs = lambda x: Variable(x, requires_grad=True)

        if not self.pretrain:
            prox_start = self.obs_acs_dim+5+1+self.k+self.hidden_dim_lstm*4
        else:
            prox_start = self.obs_acs_dim +6 + self.k
        prox_item_list = [0, self.obs_dim*self.num_agents, self.obs_dim*2*self.num_agents, self.ac_dim + self.obs_dim*2*self.num_agents]
        return ([cast_obs(self.seq_exps[:, inds, a, :self.obs_dim]) for a in range(self.num_agents)], # obs
                [cast(self.seq_exps[:, inds, a, self.obs_dim:self.obs_acs_dim]) for a in range(self.num_agents)], # actions
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim:self.obs_acs_dim+1]) for a in range(self.num_agents)], # rewards
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+1:self.obs_acs_dim+2]) for a in range(self.num_agents)], # dones
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+2:self.obs_acs_dim+3]) for a in range(self.num_agents)], # mc
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+3:self.obs_acs_dim+4]) for a in range(self.num_agents)], # n_step_targets
                [cast(self.seq_exps[:, inds, a, self.obs_acs_dim+4:self.obs_acs_dim+5]) for a in range(self.num_agents)], # ws
                self.seq_exps[0, inds, 0, self.obs_acs_dim+6+self.k:self.obs_acs_dim+6+self.k+self.hidden_dim_lstm*4], # recurrent states for both critics
                [[[cast_obs(self.seq_exps[:, inds, outer, prox_start+(inner*self.obs_dim)+prox_item_list[p]:prox_start+(inner*self.obs_dim)+prox_item_list[p]+self.obs_dim])
                if p <= 1 else cast(self.seq_exps[:, inds, outer, prox_start+(inner*self.ac_dim)+prox_item_list[p]:prox_start+(inner*self.ac_dim)+prox_item_list[p]+self.ac_dim])
                for inner in range(self.num_agents)] for p in range(len(prox_item_list))] for outer in range(self.num_agents)])
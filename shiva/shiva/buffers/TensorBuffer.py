import numpy as np
import torch
from torch.autograd import Variable

from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh

# class TensorBuffer(ReplayBuffer):

#     def __init__(self, max_size, num_agents, obs_dim, acs_dim):
#         super(TensorBuffer, self).__init__(max_size, num_agents, obs_dim, acs_dim)
#         self.obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim), requires_grad=False)
#         self.ac_buffer = torch.zeros((self.max_size, self.num_agents, acs_dim),requires_grad=False)
#         self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
#         self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim),requires_grad=False)
#         self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)

#     def push(self, exps):
#         nentries = len(exps)

#         if self.current_index + nentries > self.max_size:
#             rollover = self.max_size - self.current_index
#             self.obs_buffer = bh.roll(self.obs_buffer, rollover)
#             self.acs_buffer = bh.roll(self.acs_buffer, rollover)
#             self.rew_buffer = bh.roll(self.rew_buffer, rollover)
#             self.done_buffer = bh.roll(self.done_buffer, rollover)
#             self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)

#             self.current_index = 0
#             self.size = self.max_size

#         action_i = self.obs_dim
#         rew_i = action_i + self.acs_dim
#         done_i = rew_i+1
#         next_obs_i = done_i+1

#         self.obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, :self.obs_dim]
#         self.ac_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.ac_dim] = exps[:, :, action_i:rew_i]
#         self.rew_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, rew_i: done_i]
#         self.done_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, done_i:done_i+1]
#         self.next_obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] =  exps[:, :, next_obs_i:]

#     def sample(self, inds, to_gpu=False, device='cpu'):
#         if to_gpu:
#             cast = lambda x: Variable(x, requires_grad=False).to(device)
#             cast_obs = lambda x: Variable(x, requires_grad=True).to(device)
#         else:
#             cast = lambda x: Variable(x, requires_grad=False)
#             cast_obs = lambda x: Variable(x, requires_grad=True)

#         return (
#                     [cast_obs(self.obs_buffs[inds, i, :]) for i in range(self.num_agents)],
#                     [cast(self.ac_buffs[inds, i, :]) for i in range(self.num_agents)],
#                     [cast(self.rew_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
#                     [cast(self.done_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
#                     [cast_obs(self.next_obs_buffs[inds, i, :]) for i in range(self.num_agents)]
#                 )

class TensorBuffer(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (self.max_size, acs_dim) ,requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1), dtype=torch.bool, requires_grad=False)

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.current_index = 0
            # self.size = self.max_size
        self.obs_buffer[self.current_index:self.current_index + nentries, :self.obs_dim] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :self.acs_dim] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :1] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :1] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :self.obs_dim] = next_obs
        if self.size < self.max_size:
            self.size += nentries
            if self.size + nentries > self.max_size:
                self.size = self.max_size
        self.current_index += nentries

    def sample(self, device='cpu'):
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
                    cast_obs(self.obs_buffer[inds, :]),
                    cast(self.acs_buffer[inds, :]),
                    cast(self.rew_buffer[inds, :]).squeeze(),
                    cast_obs(self.next_obs_buffer[inds, :]),
                    cast(self.done_buffer[inds, :]).squeeze()
        )

class TensorBufferLogProbs(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorBufferLogProbs, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (self.max_size, acs_dim) ,requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.log_probs_buffer = torch.zeros( (self.max_size), requires_grad=False)

    def push(self, exps):


        obs, ac, rew, next_obs, done, log_probs = exps
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.log_probs_buffer = bh.roll(self.log_probs_buffer, rollover)

            self.current_index = 0
            # self.size = self.max_size

        # print(ac)
        # input()
        # print(log_probs)
        # input()

        self.obs_buffer[self.current_index:self.current_index+nentries, :self.obs_dim] = obs
        self.acs_buffer[self.current_index:self.current_index+nentries, :self.acs_dim] = ac
        self.rew_buffer[self.current_index:self.current_index+nentries, :1] = rew
        self.done_buffer[self.current_index:self.current_index+nentries, :1] = done
        self.next_obs_buffer[self.current_index:self.current_index+nentries, :self.obs_dim] = next_obs
        self.log_probs_buffer[self.current_index:self.current_index+nentries] = log_probs

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += 1

    def sample(self, device='cpu'):
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
                    cast_obs(self.obs_buffer[inds, :]),
                    cast(self.acs_buffer[inds, :]),
                    cast(self.rew_buffer[inds, :]).squeeze(),
                    cast_obs(self.next_obs_buffer[inds, :]),
                    cast(self.done_buffer[inds, :]).squeeze(),
                    cast(self.log_probs_buffer[inds, :])
        )

    def full_buffer(self, device='cpu'):

        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return   (
                    cast_obs(self.obs_buffer[:self.current_index,:]),
                    cast(self.acs_buffer[:self.current_index,:]),
                    cast(self.rew_buffer[:self.current_index,:]).squeeze(),
                    cast_obs(self.next_obs_buffer[:self.current_index,:]),
                    cast(self.done_buffer[:self.current_index,:]).squeeze(),
                    cast(self.log_probs_buffer[:self.current_index])
        )

    def clear_buffer(self):
        self.obs_buffer.fill_(0)
        self.acs_buffer.fill_(0)
        self.rew_buffer.fill_(0)
        self.next_obs_buffer.fill_(0)
        self.done_buffer.fill_(0)
        self.log_probs_buffer.fill_(0)
        self.current_index = 0

class TensorSingleDaggerRoboCupBuffer(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorSingleDaggerRoboCupBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, acs_dim),requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.expert_acs_buffer = torch.zeros((self.max_size, acs_dim),requires_grad=False)

    def push(self, exps):
        nentries = 1

        obs, ac, rew, next_obs, done, exp_ac = exps

        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)

            self.current_index = 0
            # self.size = self.max_size

        self.obs_buffer[self.current_index:self.current_index+nentries, :self.obs_dim] = obs
        self.acs_buffer[self.current_index:self.current_index+nentries, :self.acs_dim] = ac
        self.rew_buffer[self.current_index:self.current_index+nentries, :1] = rew
        self.done_buffer[self.current_index:self.current_index+nentries, :1] = done
        self.next_obs_buffer[self.current_index:self.current_index+nentries, :self.obs_dim] = next_obs
        self.expert_acs_buffer[self.current_index:self.current_index+nentries, :self.acs_dim] = exp_ac

        if self.size < self.max_size:
            self.size += 1
        self.current_index += 1

    def sample(self, device='cpu'):
        inds = np.random.choice(np.arange( min(len(self),self.max_size) ), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
                    cast_obs(self.obs_buffer[inds, :]),
                    cast(self.acs_buffer[inds, :]),
                    cast(self.rew_buffer[inds, :]).squeeze(),
                    cast_obs(self.next_obs_buffer[inds, :]),
                    cast(self.done_buffer[inds, :]).squeeze(),
                    cast(self.expert_acs_buffer[inds, :])
        )

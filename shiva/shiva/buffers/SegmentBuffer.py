import numpy as np
import torch
from torch.autograd import Variable
from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh

'''

    Might want to clear the buffer continually so that it is more aware of how the reward function estimator
    is changing.
    
'''


class SegmentBuffer(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(SegmentBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (self.max_size, acs_dim), requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1), requires_grad=False)

    def push(self, exps):

        obs, ac, rew, next_obs, done = exps
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index + nentries
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size

        self.obs_buffer[self.current_index:self.current_index+nentries, :self.obs_dim] = obs
        self.acs_buffer[self.current_index:self.current_index+nentries, :self.acs_dim] = ac
        self.rew_buffer[self.current_index:self.current_index+nentries, :1] = rew

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += nentries

    def sample(self, device='cpu'):
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
                    cast_obs(self.obs_buffer[inds, :]),
                    cast(self.acs_buffer[inds, :]),
                    cast(self.rew_buffer[inds, :]).squeeze(),
        )

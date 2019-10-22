import torch
import helpers.buffer_handler as bh
from torch.autograd import Variable

class TensorBuffer(ReplayBuffer):

    def __init__(self, max_size, num_agents, obs_dim, acs_dim):
        super(TensorBuffer, self).__init__(max_size, num_agents, obs_dim, acs_dim)

    def push(self, exps):
        nentries = len(exps)

        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)

            self.current_index = 0
            self.size = self.max_size

        action_i = self.obs_dim
        rew_i = action_i + self.acs_dim
        done_i = rew_i+1
        next_obs_i = done_i+1

        self.obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] = exps[:, :, :self.obs_dim]
        self.ac_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.ac_dim] = exps[:, :, action_i:rew_i]
        self.rew_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, rew_i: done_i]
        self.done_buffer[self.curr_i:self.curr_i+nentries, :self.num_agents, :1] = exps[:, :, done_i:done_i+1]
        self.next_obs_buffer[self.current_index:self.current_index+nentries, :self.num_agents, :self.obs_dim] =  exps[:, :, next_obs_i:]

    def sample(self, inds, to_gpu=False, device='cpu'):
        if to_gpu:
            cast = lambda x: Variable(x, requires_grad=False).to(device)
            cast_obs = lambda x: Variable(x, requires_grad=True).to(device)
        else:
            cast = lambda x: Variable(x, requires_grad=False)
            cast_obs = lambda x: Variable(x, requires_grad=True)

        return (
                    [cast_obs(self.obs_buffs[inds, i, :]) for i in range(self.num_agents)],
                    [cast(self.ac_buffs[inds, i, :]) for i in range(self.num_agents)],
                    [cast(self.rew_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                    [cast(self.done_buffs[inds, i, :]).squeeze() for i in range(self.num_agents)],
                    [cast_obs(self.next_obs_buffs[inds, i, :]) for i in range(self.num_agents)]
                )
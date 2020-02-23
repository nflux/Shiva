import copy
import numpy as np
import torch
from torch.autograd import Variable

from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh


class MultiAgentTensorBuffer(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(MultiAgentTensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.reset()

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        #         print("Received action:\n", ac)
        # print("Obs shape {} Acs shape {} Rew shape {} Next Obs shape {} Dones shape {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs

        # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
        # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += nentries


    def sample(self, agent_id=None, device='cpu'):
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        # if agent_id is None:
        return (
            cast_obs(self.obs_buffer[inds, :, :]),
            cast(self.acs_buffer[inds, :, :]),
            cast(self.rew_buffer[inds, :, :]),
            cast_obs(self.next_obs_buffer[inds, :, :]),
            cast(self.done_buffer[inds, :, :])
        )
        # else:
        #     return (
        #         cast_obs(self.obs_buffer[inds, agent_id, :]),
        #         cast(self.acs_buffer[inds, agent_id, :]),
        #         cast(self.rew_buffer[inds, agent_id, :]),
        #         cast_obs(self.next_obs_buffer[inds, agent_id, :]),
        #         cast(self.done_buffer[inds, agent_id, :])
        #     )

    def all(self):
        '''Returns all buffers'''
        return [
            self.obs_buffer[:self.current_index, :, :],
            self.acs_buffer[:self.current_index, :, :],
            self.rew_buffer[:self.current_index, :, :],
            self.next_obs_buffer[:self.current_index, :, :],
            self.done_buffer[:self.current_index, :, :]
        ]

    def all_numpy(self, reshape_fn=None):
        '''For data passing'''
        return copy.deepcopy([
            self.obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(np.float64),
            self.acs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(np.float64),
            self.rew_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(np.float64),
            self.next_obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(np.float64),
            self.done_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(np.bool)
        ])

    def agent_numpy(self, agent_id, reshape_fn=None):
        '''For data passing'''
        return [
            self.obs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.acs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.rew_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.next_obs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.done_buffer[:self.current_index, agent_id, :].cpu().detach().numpy()
        ]

    def reset(self):
        '''Resets the buffer parameters'''
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.current_index = 0
        self.size = 0

class TensorBuffer(ReplayBuffer):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (self.max_size, acs_dim), requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1), requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1), dtype=torch.bool, requires_grad=False)

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        # print("{} {} {} {} {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size

        self.obs_buffer[self.current_index:self.current_index + nentries, :self.obs_dim] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :self.acs_dim] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :1] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :1] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :self.obs_dim] = next_obs
        if self.size < self.max_size:
            self.size += nentries

        self.current_index += nentries

    def sample(self, device='cpu'):
        inds = np.random.choice(np.arange( len(self) ), size=self.batch_size, replace=True)
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
        self.obs_buffer = torch.zeros((num_agents,self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (num_agents,self.max_size, acs_dim) ,requires_grad=False)
        self.rew_buffer = torch.zeros((num_agents,self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros(( num_agents, self.max_size,obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((num_agents,self.max_size, 1),requires_grad=False)
        self.log_probs_buffer = torch.zeros( (num_agents ,self.max_size), requires_grad=False)

    def push(self, exps):


        obs, ac, rew, next_obs, done, log_probs = exps
        nentries = obs.size()[0]
        if self.current_index + 1 > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.log_probs_buffer = bh.roll(self.log_probs_buffer, rollover)

            self.current_index = 0
            self.size = self.max_size

        # print(ac)
        # input()
        #print(log_probs)
        #print(log_probs.size())
        #input()

        self.obs_buffer[:, self.current_index, :self.obs_dim] = obs.unsqueeze(dim=0)
        self.acs_buffer[:, self.current_index, :self.acs_dim] = ac.unsqueeze(dim=0)
        self.rew_buffer[:, self.current_index, :1] = rew.unsqueeze(dim=0)
        self.done_buffer[:, self.current_index, :1] = done.unsqueeze(dim=0)
        self.next_obs_buffer[:, self.current_index, :self.obs_dim] = next_obs.unsqueeze(dim=0)
        self.log_probs_buffer[:, self.current_index] = log_probs.squeeze(dim=-1)

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
                    cast_obs(self.obs_buffer[:,:self.current_index,:]),
                    cast(self.acs_buffer[:,:self.current_index,:]),
                    cast(self.rew_buffer[:,:self.current_index,:]).squeeze(),
                    cast_obs(self.next_obs_buffer[:,:self.current_index,:]),
                    cast(self.done_buffer[:,:self.current_index,:]).squeeze(),
                    cast(self.log_probs_buffer[:,:self.current_index])
        )

    def clear_buffer(self):
        self.obs_buffer.fill_(0)
        self.acs_buffer.fill_(0)
        self.rew_buffer.fill_(0)
        self.next_obs_buffer.fill_(0)
        self.done_buffer.fill_(0)
        self.log_probs_buffer.fill_(0)
        self.current_index = 0
        self.size = 0

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

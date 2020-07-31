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
            cast_obs(self.obs_buffer[inds, :, :].float()),
            cast(self.acs_buffer[inds, :, :].float()),
            cast(self.rew_buffer[inds, :, :].float()),
            cast_obs(self.next_obs_buffer[inds, :, :].float()),
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

    def all_numpy(self, astype=np.float64):
        '''For data passing'''
        return copy.deepcopy([
            self.obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.acs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.rew_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.next_obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.done_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype)
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


class MultiAgentTensorBuffer2(MultiAgentTensorBuffer):

    '''
        Adding manipulation for Agent IDs
    '''

    def __init__(self, *args, **kwargs):
        super(MultiAgentTensorBuffer2, self).__init__(*args, **kwargs)
        self.agent_ids = kwargs['agent_ids']

    def push(self, exps, agent_id=None):

        if agent_id is not None:
            obs, ac, rew, next_obs, done = exps
            # print("Obs shape {} Acs shape {} Rew shape {} Next Obs shape {} Dones shape {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
            nentries = len(obs)
            if self.current_index[agent_id] + nentries > self.max_size:
                rollover = self.max_size - self.current_index[agent_id]
                self.obs_buffer = bh.roll(self.obs_buffer, rollover)
                self.acs_buffer = bh.roll(self.acs_buffer, rollover)
                self.rew_buffer = bh.roll(self.rew_buffer, rollover)
                self.done_buffer = bh.roll(self.done_buffer, rollover)
                self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
                self.current_index[agent_id] = 0
                self.size[agent_id] = self.max_size
                # print("Roll!")

            self.obs_buffer[self.current_index[agent_id]:self.current_index[agent_id] + nentries, agent_id, :] = obs
            self.acs_buffer[self.current_index[agent_id]:self.current_index[agent_id] + nentries, agent_id, :] = ac
            self.rew_buffer[self.current_index[agent_id]:self.current_index[agent_id] + nentries, agent_id, :] = rew
            self.done_buffer[self.current_index[agent_id]:self.current_index[agent_id] + nentries, agent_id, :] = done
            self.next_obs_buffer[self.current_index[agent_id]:self.current_index[agent_id] + nentries, agent_id, :] = next_obs

            # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
            # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

            if self.size[agent_id] < self.max_size:
                self.size[agent_id] += nentries
            self.current_index[agent_id] += nentries

    def reset(self):
        '''Resets the buffer parameters'''
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, self.rew_dim), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.current_index = {_id:0 for _id in self.agent_ids}
        self.size = {_id:0 for _id in self.agent_ids}
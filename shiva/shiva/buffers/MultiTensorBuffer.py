import copy
import numpy as np
import torch
from torch.autograd import Variable
from random import uniform
from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh


class MultiAgentTensorBuffer(ReplayBuffer):

    def __init__(self, configs, num_agents, obs_dim, acs_dim):
        super(MultiAgentTensorBuffer, self).__init__(configs, num_agents, obs_dim, acs_dim)
        self.reset()

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        # print("Received action:\n", ac)
        # print("Obs shape {} Acs shape {} Rew shape {} Next Obs shape {} Dones shape {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
        nentries = len(obs)
        if self.current_index + nentries > self.max_size:
            rollover = self.max_size - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.td_error_buffer = bh.roll(self.td_error_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs
        td_errors = torch.ones(len(obs)) * self.td_error_buffer.max()
        self.td_error_buffer[self.current_index:self.current_index + nentries, :, :] = td_errors.reshape(-1, 1, 1)

        # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
        # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += nentries

    def sample(self, device):
        self.num_samples += 1
        self.update_epsilon_scale(self.num_samples)
        self.update_beta_scale(self.num_samples)

        if (uniform(0, 1) > self.epsilon) and self.prioritized:
            return self.prioritized_sample(device=device)
        else:
            return self.stochastic_sample(device=device)

    def stochastic_sample(self, agent_id=None, device='cpu'):
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
            cast_obs(self.obs_buffer[inds, :, :].float()),
            cast(self.acs_buffer[inds, :, :].float()),
            cast(self.rew_buffer[inds, :, :].float()),
            cast_obs(self.next_obs_buffer[inds, :, :].float()),
            cast(self.done_buffer[inds, :, :]),
            inds
        )

    def prioritized_sample(self, agent_id=None, device='cpu'):
        numerator = torch.pow(self.td_error_buffer[:self.size, :, :], self.alpha)
        denominator = torch.pow(self.td_error_buffer[:self.size, :, :], self.alpha).sum()
        probs = np.reshape((numerator/denominator).detach().numpy(), self.size)
        inds = np.random.choice(np.arange(self.size), size=self.batch_size, replace=False, p=probs)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
            cast_obs(self.obs_buffer[inds, :, :].float()),
            cast(self.acs_buffer[inds, :, :].float()),
            cast(self.rew_buffer[inds, :, :].float()),
            cast_obs(self.next_obs_buffer[inds, :, :].float()),
            cast(self.done_buffer[inds, :, :]),
            inds
        )

    def update_td_errors(self, td_errors, indeces):
        """Update the td error values for sampled experiences"""

        # Update values
        for ind, td_error in zip(indeces, td_errors):
            self.td_error_buffer[ind, :, :] = td_error.item() + self.omicron

        # Make probabilities sum to one; this might be extra as its done in the sample function as well
        numerator = self.td_error_buffer[:self.size, :, :]
        denominator = self.td_error_buffer[:self.size, :, :].sum()
        new_probs = numerator / denominator
        self.td_error_buffer[:self.size, :, :] = new_probs

    def update_epsilon_scale(self, num_samples):
            self.epsilon = self._get_epsilon_scale(num_samples)

    def _get_epsilon_scale(self, num_samples):
        return max(self.epsilon_end, self.decay_value(self.epsilon_start, self.stochastic_samples, num_samples, degree=self.epsilon_decay_degree))

    def decay_value(self, start, decay_end_step, current_step_count, degree=1):
        return start - start * ((current_step_count / decay_end_step) ** degree)

    # def is_e_greedy(self):
    #     if self.num_samples > self.stochastic_samples:
    #         return uniform(0, 1) < self.epsilon
    #     else:
    #         return True

    def update_beta_scale(self, num_samples):
            self.beta = self._get_beta_scale(num_samples)

    def _get_beta_scale(self, num_samples):
        return max(self.beta_end, self.decay_value(self.beta_start, self.beta_steps, num_samples, degree=self.beta_decay_degree))

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
        self.td_error_buffer = torch.full((self.max_size, self.num_agents, 1), fill_value=(1.0/self.max_size), dtype=torch.float64, requires_grad=False)
        self.current_index = 0
        self.size = 0


# Need a tensorbuffer just for the MPIEnv that uses max_size and batch_size
class MultiAgentSimpleTensorBuffer(ReplayBuffer):

    def __init__(self, configs, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(MultiAgentSimpleTensorBuffer, self).__init__(configs, num_agents, obs_dim, acs_dim)
        self.max_size = max_size
        self.batch_size = batch_size
        self.reset()

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        # print("Received action:\n", ac)
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
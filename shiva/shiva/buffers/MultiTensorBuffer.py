import copy
import numpy as np
import torch
from torch.autograd import Variable
from random import uniform
from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh

from typing import List, Tuple, Dict


class MultiAgentTensorBuffer(ReplayBuffer):
    """ Multi Agent Experience Replay Tensor Buffer

        A tensor based data structure encapsulating tensor buffer to store each type of data that is accumulated in
        the reinforcement learning loop. This buffer utilizes a stochastic sampling method.

    Args:
        capacity (int):
            The maximum amount of experiences to be stored in the Replay Buffer.
        batch_size (int):
            The amount of transitions to be sampled very update.
        num_agents (int):
            The number of agents that will be using the replay buffer. For memory allocation purposes.
        obd_dim (int):
            The expected observation dimension size.
        acs_dim (int):
            The expected action dimension size.
        configs (Dict):
            Global configs
    Returns:
        None
    """

    def __init__(self, capacity: int, batch_size: int, num_agents: int, obs_dim: int, acs_dim: int, configs: Dict) -> None:
        super(MultiAgentTensorBuffer, self).__init__(capacity, batch_size, num_agents, obs_dim, acs_dim, configs)
        self.reset()

    def push(self, exps: List) -> None:
        """ Used to store experiences in the Replay Buffer.

        Supports storing an entire trajectory of experiences or a single experience.

        Args:
            exps (List):
                A list of Observations, Actions, Rewards, Next Observations, and Done Flags.
        Returns:
            None
        """
        obs, ac, rew, next_obs, done = exps
        #         print("Received action:\n", ac)
        # print("Obs shape {} Acs shape {} Rew shape {} Next Obs shape {} Dones shape {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
        nentries = len(obs)
        if self.current_index + nentries > self.capacity:
            rollover = self.capacity - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.current_index = 0
            self.size = self.capacity
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs

        # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
        # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

        if self.size < self.capacity:
            self.size += nentries
        self.current_index += nentries

    def sample(self, device: str='cpu') -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """ Gets a stochastic sample from buffer.

        Args:
            device (str):
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors.
        """
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
            cast_obs(self.obs_buffer[inds, :, :].float()),
            cast(self.acs_buffer[inds, :, :].float()),
            cast(self.rew_buffer[inds, :, :].float()),
            cast_obs(self.next_obs_buffer[inds, :, :].float()),
            cast(self.done_buffer[inds, :, :])
        )

    def all(self):
        """Returns all transitions stored in every buffer.

        Used when you want tensors. Typically utilized by PPO or for passing entire episodic trajectories.

        Returns:
            A list of ordered trajectories as tensors. Typically the buffer is not full when this class method is utilized.
        """
        return [
            self.obs_buffer[:self.current_index, :, :],
            self.acs_buffer[:self.current_index, :, :],
            self.rew_buffer[:self.current_index, :, :],
            self.next_obs_buffer[:self.current_index, :, :],
            self.done_buffer[:self.current_index, :, :]
        ]

    def all_numpy(self, astype=np.float64):
        """ Returns all transitions stored in every

        Used when you want the data to be numpy arrays.

        Returns:
            A list of ordered trajectories as numpy arrays. Typically the buffer is not full when this class method is utilized.
        """
        return copy.deepcopy([
            self.obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.acs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.rew_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.next_obs_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.done_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype)
        ])

    def agent_numpy(self, agent_id, reshape_fn=None):
        """ Gives you the all the transitions in the buffer for a given agent.

        Returns:
            A list of ordered trajectories as tensors.
        """
        return [
            self.obs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.acs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.rew_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.next_obs_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.done_buffer[:self.current_index, agent_id, :].cpu().detach().numpy()
        ]

    def reset(self) -> None:
        """ Empties all the buffers and resets the buffer parameters.

        Returns:
            None
        """
        self.obs_buffer = torch.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.capacity, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.capacity, self.num_agents, 1), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.capacity, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.current_index = 0
        self.size = 0

    def get_metrics(self, *args, **kwargs):
        return []


class PrioritizedMultiAgentTensorBuffer(ReplayBuffer):

    def __init__(self, num_agents, obs_dim, acs_dim, configs):
        super(PrioritizedMultiAgentTensorBuffer, self).__init__(num_agents, obs_dim, acs_dim, configs)
        self._metrics = {}
        self._metrics = {(i*1000): [] for i in range(1, num_agents+1)}
        self.reset()

    def push(self, exps):
        obs, ac, rew, next_obs, done = exps
        # print("Received action:\n", ac)
        # print("Obs shape {} Acs shape {} Rew shape {} Next Obs shape {} Dones shape {}".format(obs.shape, ac.shape, rew.shape, next_obs.shape, done.shape))
        nentries = len(obs)
        if self.current_index + nentries > self.capacity:
            rollover = self.capacity - self.current_index
            self.obs_buffer = bh.roll(self.obs_buffer, rollover)
            self.acs_buffer = bh.roll(self.acs_buffer, rollover)
            self.rew_buffer = bh.roll(self.rew_buffer, rollover)
            self.done_buffer = bh.roll(self.done_buffer, rollover)
            self.next_obs_buffer = bh.roll(self.next_obs_buffer, rollover)
            self.td_error_buffer = bh.roll(self.td_error_buffer, rollover)
            self.current_index = 0
            self.size = self.capacity
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs
        # this will work for the case of one agent:
        #   - for multi agents pushing at the same time, it might need to explicitly state the number of agents
        #     in the second index of torch.ones; maybe something like torch.ones(len(obs),self.num_agents,1)
        # setting the td_errors to 1 upon being pushed needs to be tested but should be max priority if all the
        # td_errors are normalized in buffer.update_td_errors()
        self.td_error_buffer[self.current_index:self.current_index + nentries, :, :] = torch.ones(nentries, 1, 1)
        # td_errors =  torch.ones(len(obs), 1, 1) * self.td_error_buffer.max()

        # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
        # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

        if self.size < self.capacity:
            self.size += nentries
        self.current_index += nentries

    def sample(self, device):
        """
            Samples either stochastically or by priority depending on the config and epsilon.
        """
        self.num_samples += 1
        self.update_epsilon_scale(self.num_samples)
        self.update_beta_scale(self.num_samples)
        self._metrics[1000] += [('Prioritized_Buffer/Beta', self.beta, self.num_samples)]
        self._metrics[1000] += [('Prioritized_Buffer/Epsilon', self.epsilon, self.num_samples)]

        if (uniform(0, 1) > self.epsilon) and self.prioritized:
            return self.prioritized_sample(device=device)
        else:
            return self.stochastic_sample(device=device)

    def stochastic_sample(self, agent_id=None, device='cpu'):
        self.indeces = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
            cast_obs(self.obs_buffer[self.indeces, :, :].float()),
            cast(self.acs_buffer[self.indeces, :, :].float()),
            cast(self.rew_buffer[self.indeces, :, :].float()),
            cast_obs(self.next_obs_buffer[self.indeces, :, :].float()),
            cast(self.done_buffer[self.indeces, :, :])
        )

    def prioritized_sample(self, agent_id=None, device='cpu'):
        numerator = torch.pow(self.td_error_buffer[:self.size, :, :], self.alpha)
        denominator = torch.pow(self.td_error_buffer[:self.size, :, :], self.alpha).sum()
        probs = np.reshape((numerator/denominator).detach().numpy(), self.size)
        self.indeces = np.random.choice(np.arange(self.size), size=self.batch_size, replace=False, p=probs)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
            cast_obs(self.obs_buffer[self.indeces, :, :].float()),
            cast(self.acs_buffer[self.indeces, :, :].float()),
            cast(self.rew_buffer[self.indeces, :, :].float()),
            cast_obs(self.next_obs_buffer[self.indeces, :, :].float()),
            cast(self.done_buffer[self.indeces, :, :])
        )

    def update_td_errors(self, td_errors):
        """Update the td error values for sampled experiences"""

        # Update values
        for ind, td_error in zip(self.indeces, self.normalize_td_errors(td_errors)):
            # Adding self.omicron so that errors don't go to 0 and have a chance of being re-sampled
            self.td_error_buffer[ind, :, :] = td_error.item() + self.omicron

        # Update the probabilities
        numerator = self.td_error_buffer[:self.size, :, :]
        denominator = self.td_error_buffer[:self.size, :, :].sum()
        new_probs = numerator / denominator
        self.td_error_buffer[:self.size, :, :] = new_probs

    def normalize_td_errors(self, td_errors):
        td_errors = td_errors.abs()
        max_error = td_errors.max()
        td_errors /= max_error
        return td_errors

    @property
    def importance_sampling_weights(self):
        """ Importance Sampling """
        # Get the probabilities of the transitions that were sampled
        probs = self.td_error_buffer[self.indeces, :, :].reshape(-1).to(self.device)
        # Importance Sampling, Beta will anneal to 1 over time
        imp_samp_weights = torch.pow(1 / (self.size * probs), self.beta)
        # Grab the max importance sample weight
        max_weight = imp_samp_weights.max()
        # Divide all the weights by the max weight to normalize the weights between [0,1]
        return imp_samp_weights / max_weight

    def update_epsilon_scale(self, num_samples):
        self.epsilon = self._get_epsilon_scale(num_samples)

    def _get_epsilon_scale(self, num_samples):
        return max(self.epsilon_end, self.decay_value(self.epsilon_start, self.stochastic_samples, num_samples, degree=self.epsilon_decay_degree))

    def update_beta_scale(self, num_samples):
        self.beta = self._get_beta_scale(num_samples)

    def _get_beta_scale(self, num_samples):
        return min(self.beta_end, self.growth_value(self.beta_start, self.beta_steps, num_samples, degree=self.beta_decay_degree))

    def decay_value(self, start, decay_end_step, current_step_count, degree=1):
        return start - start * ((current_step_count / decay_end_step) ** degree)

    def growth_value(self, start, decay_end_step, current_step_count, degree=1):
        return start + start * ((current_step_count / decay_end_step) ** degree)

    def get_metrics(self, episodic, agent_id):
        return self._metrics[agent_id] if agent_id in self._metrics else []

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
        self.obs_buffer = torch.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.capacity, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.capacity, self.num_agents, 1), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.capacity, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.capacity, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.td_error_buffer = torch.full((self.capacity, self.num_agents, 1), fill_value=(1.0/self.capacity), dtype=torch.float64, requires_grad=False)
        self.current_index = 0
        self.size = 0
        self.num_samples = 0
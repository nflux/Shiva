import copy
import numpy as np
import torch
from torch.autograd import Variable

from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh

from typing import Dict, List, Tuple, Any, Union


class TensorBuffer(ReplayBuffer):
    """ Single Agent Experience Replay Tensor Buffer

        A tensor based data structure encapsulating tensor buffer to store each type of data that is accumulated in
        the reinforcement learning loop. This buffer utilizes a stochastic sampling method.

    Args:
        max_size (int):
            The maximum amount of experiences to be stored in the Replay Buffer.
        batch_size (int):
            The amount of transitions to be sampled very update.
        num_agents:
            The number of agents that will be using the replay buffer. For memory allocation purposes.
        obd_dim:
            The expected observation dimension size.
        acs_dim:
            The expected action dimension size.
    Returns:
        None
    """

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (self.max_size, acs_dim), requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1), requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1), dtype=torch.bool, requires_grad=False)

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

    def sample(self, device='cpu') -> Tuple:
        """ Gets a stochastic sample from buffer.
        Args:
            agent_id (int):
                Used to specify what agent's buffer to sample from.
            device (str):
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors.
        """
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
    """ Used to store experiences in the Replay Buffer.

    Supports storing an entire trajectory of experiences or a single experience. Has an additional buffer for
    storing log probabilities.

    Used for PPO.

    Args:
        exps (List):
            A list of Observations, Actions, Rewards, Next Observations, and Done Flags.
    Returns:
        None
    """

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorBufferLogProbs, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((num_agents,self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros( (num_agents,self.max_size, acs_dim) ,requires_grad=False)
        self.rew_buffer = torch.zeros((num_agents,self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros(( num_agents, self.max_size,obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((num_agents,self.max_size, 1),requires_grad=False)
        self.log_probs_buffer = torch.zeros( (num_agents ,self.max_size), requires_grad=False)

    def push(self, exps) -> None:
        """ Used to store experiences in the Replay Buffer.

        Supports storing an entire trajectory of experiences or a single experience.

        Args:
            exps (List):
                A list of Observations, Actions, Rewards, Next Observations, Done Flags and Log Probabilities.
        Returns:
            None
        """
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

        self.obs_buffer[:, self.current_index, :self.obs_dim] = obs.unsqueeze(dim=0)
        self.acs_buffer[:, self.current_index, :self.acs_dim] = ac.unsqueeze(dim=0)
        self.rew_buffer[:, self.current_index, :1] = rew.unsqueeze(dim=0)
        self.done_buffer[:, self.current_index, :1] = done.unsqueeze(dim=0)
        self.next_obs_buffer[:, self.current_index, :self.obs_dim] = next_obs.unsqueeze(dim=0)
        self.log_probs_buffer[:, self.current_index] = log_probs.squeeze(dim=-1)

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += 1

    def sample(self, device='cpu') -> Tuple:
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
                    cast_obs(self.obs_buffer[inds, :]),
                    cast(self.acs_buffer[inds, :]),
                    cast(self.rew_buffer[inds, :]).squeeze(),
                    cast_obs(self.next_obs_buffer[inds, :]),
                    cast(self.done_buffer[inds, :]).squeeze(),
                    cast(self.log_probs_buffer[inds, :])
        )

    def full_buffer(self, device='cpu') -> Tuple:
        """ Returns everything stored in the buffer.
        Args:
            device (str):
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors.
        """
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        return (
                    cast_obs(self.obs_buffer[:,:self.current_index,:]),
                    cast(self.acs_buffer[:,:self.current_index,:]),
                    cast(self.rew_buffer[:,:self.current_index,:]).squeeze(),
                    cast_obs(self.next_obs_buffer[:,:self.current_index,:]),
                    cast(self.done_buffer[:,:self.current_index,:]).squeeze(),
                    cast(self.log_probs_buffer[:,:self.current_index])
        )

    def clear_buffer(self) -> None:
        """ Empties all the buffers and resets the buffer parameters.

        Returns:
            None
        """
        self.obs_buffer.fill_(0)
        self.acs_buffer.fill_(0)
        self.rew_buffer.fill_(0)
        self.next_obs_buffer.fill_(0)
        self.done_buffer.fill_(0)
        self.log_probs_buffer.fill_(0)
        self.current_index = 0
        self.size = 0


class TensorSingleDaggerRoboCupBuffer(ReplayBuffer):
    """ Used to store experiences in the Replay Buffer.

    Supports storing an entire trajectory of experiences or a single experience. Has an additional buffer for
    storing log probabilities.

    Used for PPO.

    Args:
        exps (List):
            A list of Observations, Actions, Rewards, Next Observations, and Done Flags.
    Returns:
        None
    """
    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        super(TensorSingleDaggerRoboCupBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.obs_buffer = torch.zeros((self.max_size, obs_dim), requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, acs_dim),requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, 1),requires_grad=False)
        self.expert_acs_buffer = torch.zeros((self.max_size, acs_dim),requires_grad=False)

    def push(self, exps):
        """ Used to store experiences in the Replay Buffer.

        Supports storing an entire trajectory of experiences or a single experience.

        Args:
            exps (List):
                A list of Observations, Actions, Rewards, Next Observations, and Done Flags.
        Returns:
            None
        """
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
        """ Gets a stochastic sample from buffer.
        Args:
            device (str):
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors.
        """
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

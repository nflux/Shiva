import copy, random
import numpy as np
import torch
from torch.autograd import Variable

from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh
from shiva.helpers.segment_tree import SumSegmentTree, MinSegmentTree

class MultiAgentTensorBuffer(ReplayBuffer):
    """ Multi Agent Experience Replay Tensor Buffer

        A tensor based data structure encapsulating tensor buffer to store each type of data that is accumulated in
        the reinforcement learning loop. This buffer utilizes a stochastic sampling method.

    Args:
        max_size (int):
            The maximum amount of experiences to be stored in the Replay Buffer.
        batch_size (int):
            The amount of transitions to be sampled very update.
        num_agents:
            The number of agents that will be using the replay buffer. For memory allocation purposes.
        obs_dim:
            The expected observation dimension size.
        acs_dim:
            The expected action dimension size.
    Returns:
        None
    """
    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim, *args, **kwargs):
        super(MultiAgentTensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.reset()

    def push(self, exps):
        """ Used to store experiences in the Replay Buffer.

        Supports storing an entire trajectory of experiences or a single experience.

        Args:
            exps (List):
                A list of Observations, Actions, Rewards, Next Observations, Done Flags, Current state action mask, Next state action mask
        Returns:
            None
        """
        obs, ac, rew, next_obs, done, acs_mask, next_acs_mask = exps
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
            self.acs_mask_buffer = bh.roll(self.acs_mask_buffer, rollover)
            self.next_acs_mask_buffer = bh.roll(self.next_acs_mask_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs
        self.acs_mask_buffer[self.current_index:self.current_index + nentries, :, :] = acs_mask
        self.next_acs_mask_buffer[self.current_index:self.current_index + nentries, :, :] = next_acs_mask

        # print("From in-buffer Obs {}".format(self.obs_buffer[self.current_index:self.current_index + nentries, :, :]))
        # print("From in-buffer Acs {}".format(self.acs_buffer[self.current_index:self.current_index + nentries, :, :]))

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += nentries

    def sample(self, agent_id=None, device='cpu'):
        """ Gets a stochastic sample from buffer.
        Args:
            agent_id (int):
                Used to specify what agent's buffer to sample from.
            device (str):
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors.
        """
        inds = np.random.choice(np.arange(len(self)), size=self.batch_size, replace=True)
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        # if agent_id is None:
        return (
            cast_obs(self.obs_buffer[inds, :, :].float()),
            cast(self.acs_buffer[inds, :, :].float()),
            cast(self.rew_buffer[inds, :, :].float()),
            cast_obs(self.next_obs_buffer[inds, :, :].float()),
            cast(self.done_buffer[inds, :, :]),
            cast(self.acs_mask_buffer[inds, :, :]),
            cast(self.next_acs_mask_buffer[inds, :, :])
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
            self.done_buffer[:self.current_index, :, :],
            self.acs_mask_buffer[:self.current_index, :, :],
            self.next_acs_mask_buffer[:self.current_index, :, :]
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
            self.done_buffer[:self.current_index, :, :].cpu().detach().numpy().astype(astype),
            self.acs_mask_buffer[:self.current_index, :, :].cpu().detach().numpy(),
            self.next_acs_mask_buffer[:self.current_index, :, :].cpu().detach().numpy()
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
            self.done_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.acs_mask_buffer[:self.current_index, agent_id, :].cpu().detach().numpy(),
            self.next_acs_mask_buffer[:self.current_index, agent_id, :].cpu().detach().numpy()
        ]

    def reset(self):
        """ Empties all the buffers and resets the buffer parameters.

        Returns:
            None
        """
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.acs_mask_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.bool, requires_grad=False)
        self.next_acs_mask_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.bool, requires_grad=False)
        self.current_index = 0
        self.size = 0


class MultiAgentPrioritizedTensorBuffer(ReplayBuffer):
    """ Multi Agent Experience Replay Tensor Buffer

        A tensor based data structure encapsulating tensor buffer to store each type of data that is accumulated in
        the reinforcement learning loop. This buffer utilizes a stochastic sampling method.

    Args:
        max_size (int):
            The maximum amount of experiences to be stored in the Replay Buffer.
        batch_size (int):
            The amount of transitions to be sampled very update.
        num_agents:
            The number of agents that will be using the replay buffer. For memory allocation purposes.
        obs_dim:
            The expected observation dimension size.
        acs_dim:
            The expected action dimension size.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
    Returns:
        None
    """
    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim, alpha, *args, **kwargs):
        super(MultiAgentPrioritizedTensorBuffer, self).__init__(max_size, batch_size, num_agents, obs_dim, acs_dim)
        self.prioritized = True
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < max_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.reset()

    def push(self, exps):
        """ Used to store experiences in the Replay Buffer.

        Supports storing an entire trajectory of experiences or a single experience.

        Args:
            exps (List):
                A list of Observations, Actions, Rewards, Next Observations, Done Flags, Current state action mask, Next state action mask
        Returns:
            None
        """
        obs, ac, rew, next_obs, done, acs_mask, next_acs_mask = exps
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
            self.acs_mask_buffer = bh.roll(self.acs_mask_buffer, rollover)
            self.next_acs_mask_buffer = bh.roll(self.next_acs_mask_buffer, rollover)
            self.current_index = 0
            self.size = self.max_size
            # print("Roll!")

        self.obs_buffer[self.current_index:self.current_index + nentries, :, :] = obs
        self.acs_buffer[self.current_index:self.current_index + nentries, :, :] = ac
        self.rew_buffer[self.current_index:self.current_index + nentries, :, :] = rew
        self.done_buffer[self.current_index:self.current_index + nentries, :, :] = done
        self.next_obs_buffer[self.current_index:self.current_index + nentries, :, :] = next_obs
        self.acs_mask_buffer[self.current_index:self.current_index + nentries, :, :] = acs_mask
        self.next_acs_mask_buffer[self.current_index:self.current_index + nentries, :, :] = next_acs_mask

        # Set maximum priority for each of the new entries
        for idx in range(nentries):
            self._it_sum[self.current_index+idx] = self._max_priority ** self._alpha
            self._it_min[self.current_index+idx] = self._max_priority ** self._alpha

        if self.size < self.max_size:
            self.size += nentries
        self.current_index += nentries

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, beta=None, agent_id=None, device='cpu'):
        """
        Args:
            beta: float
                To what degree to use importance weights
                (0 - no corrections, 1 - full correction)
            agent_id: int
                Used to specify what agent's buffer to sample from.
            device: str
                Used to specify if you want the sample to placed on cpu or gpu.

        Returns:
            A tuple of tensors (obs, acs, rew, next_obs, done, acs_mask, next_acs_mask, weights, idxs)
        """
        beta = self._beta if beta is None else beta
        cast = lambda x: Variable(x, requires_grad=False).to(device)
        cast_obs = lambda x: Variable(x, requires_grad=True).to(device)

        idxes = self._sample_proportional(self.batch_size)

        ws = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.size) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            w = (p_sample * self.size) ** (-beta)
            ws.append(w / max_weight)
        weights = np.array(ws)

        # if agent_id is None:
        print(len(idxes), type(idxes))
        return (
            cast_obs(self.obs_buffer[idxes, :, :].float()),
            cast(self.acs_buffer[idxes, :, :].float()),
            cast(self.rew_buffer[idxes, :, :].float()),
            cast_obs(self.next_obs_buffer[idxes, :, :].float()),
            cast(self.done_buffer[idxes, :, :]),
            cast(self.acs_mask_buffer[idxes, :, :]),
            cast(self.next_acs_mask_buffer[idxes, :, :]),
            cast(torch.from_numpy(weights)),
            cast(torch.tensor(idxes))
        )
        # else:
        #     return (
        #         cast_obs(self.obs_buffer[inds, agent_id, :]),
        #         cast(self.acs_buffer[inds, agent_id, :]),
        #         cast(self.rew_buffer[inds, agent_id, :]),
        #         cast_obs(self.next_obs_buffer[inds, agent_id, :]),
        #         cast(self.done_buffer[inds, agent_id, :])
        #     )

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def reset(self):
        """ Empties all the buffers and resets the buffer parameters.

        Returns:
            None
        """
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.acs_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.float64, requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.float64, requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), dtype=torch.float64, requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1), dtype=torch.bool, requires_grad=False)
        self.acs_mask_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.bool, requires_grad=False)
        self.next_acs_mask_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim), dtype=torch.bool, requires_grad=False)
        self.current_index = 0
        self.size = 0

    def set_beta(self, beta):
        self._beta = beta
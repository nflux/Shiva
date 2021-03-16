import copy
import numpy as np
import torch
from torch.autograd import Variable

from shiva.buffers.ReplayBuffer import ReplayBuffer
from shiva.helpers import buffer_handler as bh


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
    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
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

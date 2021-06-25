import torch


class ReplayBuffer(object):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        """ Abstract Replay Buffer class all environments implemented in Shiva inherit from."""
        self.current_index = 0
        self.size = 0
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.rew_dim = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prioritized = False

    def __len__(self):
        return self.size

    def push(self) -> None:
        """Pushes experiences into the buffer."""

    def sample(self) -> None:
        """Samples experiences from the buffer."""

    def clear(self) -> None:
        """ Clears the buffer."""

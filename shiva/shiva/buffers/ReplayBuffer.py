import torch
from abc import ABC, abstractmethod
from typing import Dict


class ReplayBuffer(ABC):
    """ Abstract Replay Buffer class all environments implemented in Shiva inherit from."""
    def __init__(self, capacity: int, batch_size: int, num_agents: int, obs_dim: int, acs_dim: int, configs: Dict):
        {setattr(self, k, v) for k, v in configs['Buffer'].items()}
        self.configs = configs
        self.prioritized = self.prioritized if hasattr(self, 'prioritized') else False # for global support on non-prioritized buffer
        self.current_index = 0
        self.size = 0
        self.num_samples = 0
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.rew_dim = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def push(self, *args, **kwargs):
        """Pushes experiences into the buffer."""
        raise NotImplementedError("To be implemented by subclass")

    def sample(self, *args, **kwargs):
        """Samples experiences from the buffer."""
        raise NotImplementedError("To be implemented by subclass")

    def clear(self, *args, **kwargs):
        """ Clears the buffer."""
        raise NotImplementedError("To be implemented by subclass")

    def get_metrics(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by subclass")
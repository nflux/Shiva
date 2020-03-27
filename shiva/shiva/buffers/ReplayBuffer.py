import torch
import numpy as np

class ReplayBuffer(object):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        self.current_index = 0
        self.size = 0
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = np.random.randint(0, 100)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def __len__(self):
        return self.size

    def push(self):
        assert "NotImplemented"

    def sample(self):
        assert "NotImplemented"

    def clear(self):
        assert "NotImplemented"

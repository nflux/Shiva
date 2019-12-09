import torch

class ReplayBuffer(object):

    def __init__(self, max_size, batch_size, num_agents, obs_dim, acs_dim):
        self.current_index = 0
        self.size = 0
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim

    def __len__(self):
        return self.size

    def push(self):
        pass

    def sample(self):
        pass

    def clear(self):
        pass
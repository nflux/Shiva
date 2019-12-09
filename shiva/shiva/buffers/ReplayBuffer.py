import torch

class ReplayBuffer(object):

    def __init__(self, max_size, num_agents, obs_dim, acs_dim):
        self.current_index = 0
        self.size = 0
        self.num_agents = num_agents
        self.max_size = max_size
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim), requires_grad=False)
        self.ac_buffer = torch.zeros((self.max_size, self.num_agents, acs_dim),requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim

    def __len__(self):
        return self.size

    def push(self):
        pass

    def sample(self):
        pass

    def clear(self):
        self.current_index = 0
        self.size = 0
        self.obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim), requires_grad=False)
        self.ac_buffer = torch.zeros((self.max_size, self.num_agents, self.acs_dim),requires_grad=False)
        self.rew_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
        self.next_obs_buffer = torch.zeros((self.max_size, self.num_agents, self.obs_dim),requires_grad=False)
        self.done_buffer = torch.zeros((self.max_size, self.num_agents, 1),requires_grad=False)
import torch


class ReplayBuffer(object):

    def __init__(self, num_agents, obs_dim, acs_dim, configs):
        assert 'capacity' in configs['Buffer'], "Buffer needs 'capacity' attribute"
        assert 'batch_size' in configs['Buffer'], "Buffer needs 'batch_size' attribute"
        {setattr(self, k, v) for k, v in configs['Buffer'].items()}
        self.prioritized = self.prioritized if hasattr(self, 'prioritized') else False # for global support on non-prioritized buffer
        self.current_index = 0
        self.size = 0
        self.num_samples = 0
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.rew_dim = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def push(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by subclass")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by subclass")

    def clear(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by subclass")

    def get_metrics(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by subclass")
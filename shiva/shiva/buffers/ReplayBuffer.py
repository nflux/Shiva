import torch


class ReplayBuffer(object):

    def __init__(self, configs, num_agents, obs_dim, acs_dim):
        try:
            for k, v in configs['Buffer'].items():
                setattr(self, k, v)
        except:
            pass
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

    def push(self):
        assert "NotImplemented"

    def sample(self):
        assert "NotImplemented"

    def clear(self):
        assert "NotImplemented"
import numpy as np
import torch

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

    def set_scale(self, scale):
        self.scale = scale


class OUNoiseTorch:
    def __init__(self, output_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self._output_dim = (output_dimension,)
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(self._output_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self._output_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self._output_dim)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).clone()

    def set_output_dim(self, output_dim):
        if self._output_dim != output_dim:
            self._output_dim = output_dim
            self.reset()

    def set_scale(self, scale):
        self.scale = scale
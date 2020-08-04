import numpy as np
import torch

class OUNoise:
    def __init__(self, action_dimension: int, scale: float=0.1, mu: float=0, theta: float=0.15, sigma: float=0.2):
        """
        OUNoise using numpy and following https://github.com/songrotek/DDPG/blob/master/ou_noise.py

        Args:
            action_dimension (int):
            scale (float):
            mu (float):
            theta (float):
            sigma (float):
        """
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self) -> None:
        """
        Resets the state of the OUNoise

        Returns:
            None
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self) -> np.ndarray:
        """
        Returns a numpy array containing noise values

        Returns:
            np.ndarray
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

    def set_scale(self, scale: float) -> None:
        """
        Change the scale of the noise

        Args:
            scale (float): new scale for the noise

        Returns:
            None
        """
        self.scale = scale


class OUNoiseTorch:
    def __init__(self, output_dimension: int, scale: float=0.1, mu: float=0, theta: float=0.15, sigma: float=0.2):
        """
        Same as OUNoise but using torch tensors

        Args:
            output_dimension (int):
            scale (float):
            mu (float):
            theta (float):
            sigma (float):
        """
        self._output_dim = (output_dimension,)
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(self._output_dim) * self.mu
        self.reset()

    def reset(self) -> None:
        """
        Resets the state of the OUNoise

        Returns:
            None
        """
        self.state = torch.ones(self._output_dim) * self.mu

    def noise(self) -> torch.tensor:
        """
        Returns a torch tensor containing noise values

        Returns:
            torch.tensor
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self._output_dim)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).clone()

    def set_output_dim(self, output_dim: int) -> None:
        """
        Sets the output dimension of the noise

        Args:
            output_dim: new output dimension

        Returns:
            None
        """
        if self._output_dim != output_dim:
            self._output_dim = output_dim
            self.reset()

    def set_scale(self, scale: float) -> None:
        """
        Sets the scale for the noise

        Args:
            scale (float): new scale

        Returns:
            None
        """
        self.scale = scale
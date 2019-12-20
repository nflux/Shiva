from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch

from shiva.buffers.SimpleBuffer import SimpleBuffer
from shiva.buffers.TensorBuffer import TensorBuffer

class test_buffers(TestCase):

    def tearDown(self):
        pass

    def setUp(self):
        self.observation_space = 10 # np.random.randint(20)
        self.action_space = 3 # np.random.randint(10)
        self.n_agents = 1 # np.random.randint(5)
        self.capacity = 100 # np.random.randint(128)
        self.batch_size = 10 # np.random.randint(32)
        self.setup_numpys()
        self.setup_tensors()

    def setup_numpys(self):
        self.observations = np.random.rand(self.n_agents, self.observation_space)
        self.actions = np.random.rand(self.n_agents, self.action_space)
        self.rewards = np.random.rand(1, self.n_agents)
        self.next_observations = np.random.rand(self.n_agents, self.observation_space)
        self.dones = np.random.randint(0, 2, (1, self.n_agents))
        # self.experiences = [self.observations, self.actions, self.rewards, self.next_observations, self.dones]

    def setup_tensors(self):
        self.observations_t = np.random.rand(self.n_agents, self.observation_space)
        self.actions_t = torch.rand(self.n_agents, self.action_space)
        self.rewards_t = torch.rand(1, self.n_agents)
        self.next_observations_t = torch.rand(self.n_agents, self.observation_space)
        self.dones_t = torch.randint(0, 2, (1, self.n_agents))
        # self.experiences = [self.observations, self.actions, self.rewards, self.next_observations, self.dones]

    def test_simple_buffer(self):
        buffer = SimpleBuffer(self.batch_size, self.capacity)
        assert len(buffer) == 0, 'equal zero'
        # try inserting
        for obs, act, rew, next_obs, done in zip(self.observations, self.actions, self.rewards, self.next_observations, self.dones):
            buffer.append([obs, act, rew, next_obs, done])
        # try sampling sizes
        obs, act, rew, next_obs, dones = buffer.sample()
        assert obs.shape == (self.batch_size, self.observations.shape[1])
        assert act.shape == (self.batch_size, self.actions.shape[1])
        assert rew.shape == (self.batch_size, self.rewards.shape[1])
        assert next_obs.shape == (self.batch_size, self.next_observations.shape[1])
        assert dones.shape == (self.batch_size, self.dones.shape[1])

    def test_tensor_buffer(self):
        buffer = TensorBuffer
from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch

from shiva.algorithms.DQNAlgorithm import DQNAlgorithm
from shiva.buffers.SimpleBuffer import SimpleBuffer
from shiva.agents.DQNAgent import DQNAgent
from shiva.helpers.config_handler import load_class, load_config_file_2_dict
from shiva.helpers.dir_handler import find_pattern_in_path

class test_algorithms(TestCase):
    templates_dir = './configs/Templates/'
    config_ext = '.ini'

    def tearDown(self):
        pass

    def setUp(self):
        self.templates = find_pattern_in_path(self.templates_dir, self.config_ext)

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

    def setup_tensors(self):
        self.observations_t = np.random.rand(self.n_agents, self.observation_space)
        self.actions_t = torch.rand(self.n_agents, self.action_space)
        self.rewards_t = torch.rand(1, self.n_agents)
        self.next_observations_t = torch.rand(self.n_agents, self.observation_space)
        self.dones_t = torch.randint(0, 2, (1, self.n_agents))

    def set_agent(self):
        cls = load_class('shiva.agents', self.config['Agent']['type'])
        self.agent = cls(0, self.action_space, self.observation_space, self.config['Agent'], self.config['Network'])

    def set_buffer(self):
        cls = load_class('shiva.buffers', self.config['Buffer']['type'])
        self.buffer = cls(self.batch_size, self.capacity)
        for obs, act, rew, next_obs, done in zip(self.observations, self.actions, self.rewards, self.next_observations, self.dones):
            self.buffer.append([obs, act, rew, next_obs, done])

    def set_algorithm(self):
        cls = load_class('shiva.algorithms', self.config['Algorithm']['type'])
        self.alg = cls(self.observation_space, self.action_space, self.config)

    def test_algorithms(self):
        for template in self.templates:
            self.config = load_config_file_2_dict(template)
            self.set_agent()
            self.set_buffer()
            self.set_algorithm()
            self.testing()

    def testing(self):
        try:
            self.alg.update(self.agent, self.buffer, 1000)
        except:
            self.alg.update(self.agent, self.buffer.sample(), 1000)
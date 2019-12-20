from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import torch
from copy import deepcopy

import logging
# logging.basicConfig(filename='example.log', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

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

    def set_buffer(self):
        cls = load_class('shiva.buffers', self.config['Buffer']['type'])
        self.buffer = cls(self.batch_size, self.capacity)
        for obs, act, rew, next_obs, done in zip(self.observations, self.actions, self.rewards, self.next_observations, self.dones):
            self.buffer.append([obs, act, rew, next_obs, done])

    def set_algorithm(self):
        cls = load_class('shiva.algorithms', self.config['Algorithm']['type'])
        try:
            self.alg = cls(self.observation_space, self.action_space, [self.config['Algorithm'], self.config['Agent'], self.config['Network']])
        except:
            # PPO
            self.alg = cls(self.observation_space, self.action_space, None, 1, [self.config['Algorithm'], self.config['Agent'], self.config['Network']])

    def set_agent(self):
        self.agent = self.alg.create_agent()

    def test_algorithms(self):
        for template in self.templates:
            self.config = load_config_file_2_dict(template)
            LOGGER.info('Algorithm test for {}'.format(self.config['Algorithm']['type']))
            self.config['Admin']['Save'] = False
            self.set_buffer()
            self.set_algorithm()
            self.set_agent()
            try:
                # DQN and DDPG
                self.alg.update(self.agent, self.buffer.sample(), 1000)
            except:
                try:
                    # PPO to fix, put the extra agent on the algorithm
                    self.agent2 = deepcopy(self.agent)
                    self.alg.update(self.agent, self.agent2, self.buffer.sample(), 1000)
                except:
                    # TD3
                    self.alg.update(self.agent, self.buffer, 1000)
            LOGGER.info('\tUpdate() OK')
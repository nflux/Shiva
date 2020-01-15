import torch
import copy
import random
import numpy as np

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class DummyLearner(Learner):
    def __init__(self, learner_id, config, port=None):
        super(DummyLearner ,self).__init__(learner_id, config, port)

    def run(self, train=True):
        self.step_count_per_run = 0
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()
        self.env.close()

    def step(self):
        # observation = self.env.get_observation()
        left_next_observation, left_reward, right_next_observation, right_reward, done, more_data = self.env.step(left_actions=self.left_dummy_actions, 
                                                                right_actions=self.right_dummy_actions, 
                                                                discrete_select='argmax', collect=False, device=self.device)
    
    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs, self.port)

    def launch(self):
        self.env = self.create_environment()
        # Turn in Circles constantly
        left_actions = [[0, 1, 0, 0, 0, 1, 0, 0] for _ in range(self.env.num_left)]
        right_actions = [[0, 1, 0, 0, 0, 1, 0, 0] for _ in range(self.env.num_right)]
        self.left_dummy_actions = torch.tensor(left_actions).float().to(self.device)
        self.right_dummy_actions = torch.tensor(right_actions).float().to(self.device)
        print('Launch Successful.')
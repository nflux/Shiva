import torch
import copy
import random
import numpy as np

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentTD3Learner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentTD3Learner, self).__init__(learner_id, config)

    def run(self, train=True):
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()
                self.alg.update(self.agent, self.buffer, self.env.step_count)
                self.collect_metrics()
            self.alg.update(self.agent, self.buffer, self.env.step_count, episodic=True)
            self.collect_metrics(episodic=True)
            self.checkpoint()
            print('Episode {} complete on {} steps!\tEpisodic reward: {} '.format(self.env.done_count, self.env.steps_per_episode, self.env.reward_per_episode))
        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        """Temporary fix for Unity as it receives multiple observations"""
        if len(observation.shape) > 1:
            action = [self.agent.get_action(obs, self.env.step_count) for obs in observation]
            next_observation, reward, done, more_data = self.env.step(action)
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                self.buffer.push(list(map(torch.clone, (torch.tensor(obs), torch.from_numpy(act), torch.tensor(rew), torch.tensor(next_obs), torch.tensor([don], dtype=torch.bool)))))
        else:
            action = self.agent.get_action(observation, self.env.step_count)
            next_observation, reward, done, more_data = self.env.step(action)
            self.buffer.push(list(map(torch.clone, (torch.tensor(observation), torch.tensor(action), torch.tensor(reward), torch.tensor(next_observation), torch.tensor([done], dtype=torch.bool)))))
        """"""

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(), self.configs)

    def create_buffer(self):
        # buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        # return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], 1, self.env.get_observation_space(), self.env.get_action_space()['acs_space'])

    def launch(self):
        self.env = self.create_environment()
        self.alg = self.create_algorithm()
        if self.load_agents:
            self.agent = Admin._load_agent(self.load_agents)
            self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent(self.get_new_agent_id())
            if self.using_buffer:
                self.buffer = self.create_buffer()
        print('Launch Successful.')
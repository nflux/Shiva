from __main__ import shiva
from .Learner import Learner
import helpers.misc as misc
# import torch.multiprocessing as mp
import torch
import envs
import algorithms
import buffers
import copy
import random
import numpy as np

class SingleAgentTD3Learner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentTD3Learner ,self).__init__(learner_id, config)
        np.random.seed(self.configs['Algorithm']['manual_seed'])
        torch.manual_seed(self.configs['Algorithm']['manual_seed'])

    def run(self):
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()
                self.alg.update(self.agent, self.buffer, self.env.step_count)
                self.collect_metrics(episodic=False)
            self.collect_metrics(episodic=True)
            print('Episode {} complete on {} steps!\tEpisodic reward: {} '.format(self.env.done_count, self.env.steps_per_episode, self.env.reward_per_episode))
        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        """Temporary fix for Unity as it receives multiple observations"""
        if len(observation.shape) > 1:
            action = [self.alg.get_action(self.agent, obs, self.env.step_count) for obs in observation]
            next_observation, reward, done, more_data = self.env.step(action)
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                # print(act, rew, don)
                self.buffer.append(exp)
        else:
            action = self.alg.get_action(self.agent, observation, self.env.step_count)
            next_observation, reward, done, more_data = self.env.step(action)
            t = [observation, action, reward, next_observation, int(done)]
            exp = copy.deepcopy(t)
            self.buffer.append(exp)
        """"""

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer = getattr(buffers,self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agent

    def get_algorithm(self):
        return self.alg

    def launch(self):
        self.env = self.create_environment()
        self.alg = self.create_algorithm()
        if self.load_agents:
            self.agent = self.load_agent(self.load_agents)
            self.alg.agent = self.agent
        else:
            self.agent = self.alg.create_agent(self.get_id())

        if self.using_buffer:
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agent(path)
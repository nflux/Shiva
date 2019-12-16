from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import torch.multiprocessing as mp
import envs
import algorithms
import buffers
import copy
import random
import numpy as np

class SingleAgentLearner(Learner):
    '''
        Start idea of a Single Agent Learner being able to play on Unity and Gym 
            - potentially being able to play on RoboCup as well

    '''

    def __init__(self, learner_id, config):
        super(SingleAgentLearner,self).__init__(learner_id, config)

    def run(self):
        self.step_count = 0
        # for self.ep_count in range(self.episodes):
        while self.env.finished(self.episodes):
            self.env.reset()
            self.totalReward = 0
            # done = False
            # while not done:
            while not self.env.finished(1): # play one episode
                # done = self.step()
                self.step()
                self.collect_metrics() # metrics per step
                self.step_count += 1
            self.collect_metrics(True) # metrics per episode
            print('Step: {} episode complete!'.format(self.ep_count))
            if self.ep_count % self.configs['Algorithm']['update_episodes'] == 0:
                self.alg.update(self.agent, self.old_agent, self.buffer.sample(), self.step_count)
                self.buffer.clear_buffer()
                self.old_agent = copy.deepcopy(self.agent)
        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        action = self.agent.get_action(observation)

        next_observation, reward, done, more_data = self.env.step(action)

        """Temporary fix for Unity as it receives multiple observations"""
        if len(observation.shape) > 1:
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                exp = copy.deepcopy(exp)
                self.buffer.append(exp)
        else:
            t = [observation, action, reward, next_observation, int(done)]
            deep = copy.deepcopy(t)
            self.buffer.append(deep)
        """"""

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        acs_continuous = self.env.action_space_continuous
        acs_discrete= self.env.action_space_discrete
        return algorithm(self.env.get_observation_space(), self.env.get_action_space(), acs_discrete, acs_continuous, [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer = getattr(buffers,self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):

        # Launch the environment
        self.env = self.create_environment()

        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()
        # Create the agent
        if self.load_agents:
            self.agent= self.load_agent(self.load_agents)
        else:
            self.agent= self.alg.create_agent()
            self.old_agent = self.alg.create_agent()
            self.old_agent = copy.deepcopy(self.agent)


        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]

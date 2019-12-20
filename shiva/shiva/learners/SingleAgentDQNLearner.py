import numpy as np
import random
import copy

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentDQNLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentDQNLearner,self).__init__(learner_id, config)

    def run(self, train=True):
        self.step_count = 0
        # for self.ep_count in range(self.episodes):
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()
                self.step_count += 1
                self.collect_metrics() # metrics per step
            self.ep_count += 1
            self.collect_metrics(True) # metrics per episode
            self.alg.update(self.agent, self.buffer.sample(), self.step_count)
            print('Episode {} complete on {} steps!\tEpisodic reward: {} '.format(self.env.done_count, self.env.steps_per_episode, self.env.get_total_reward()))
        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        """Temporary fix for Unity as it receives multiple observations"""
        if len(observation.shape) > 1:
            action = [self.alg.get_action(self.agent, obs, self.step_count) for obs in observation]
            next_observation, reward, done, more_data = self.env.step(action)
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                # print(act, rew, don)
                self.buffer.append(exp)
        else:
            action = self.alg.get_action(self.agent, observation, self.step_count)
            next_observation, reward, done, more_data = self.env.step(action)
            t = [observation, action, reward, next_observation, int(done)]
            exp = copy.deepcopy(t)
            self.buffer.append(exp)
        """"""

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        try:
            self.configs['Agent']['learning_rate'] = random.uniform(self.learning_rate[0],self.learning_rate[1])
            return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        except:
            return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),[self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def launch(self):
        self.env = self.create_environment()

        self.alg = self.create_algorithm()

        if self.load_agents:
            self.agent = Admin._load_agents(self.load_agents)[0]
            self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent()
            if self.using_buffer:
                self.buffer = self.create_buffer()

        print('Launch Successful.')

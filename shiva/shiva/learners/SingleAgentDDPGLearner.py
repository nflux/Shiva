import numpy as np
import copy
import time

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentDDPGLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentDDPGLearner,self).__init__(learner_id, config)
        np.random.seed(5)

    def run(self):
        # self.step_count = 0
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()
                self.alg.update(self.agent, self.buffer, self.env.step_count)
                self.collect_metrics()  # metrics per episode
            self.alg.update(self.agent, self.buffer, self.env.step_count, episodic=True)
            self.collect_metrics(episodic=True)  # metrics per episode
            self.checkpoint()
        self.env.close()

    def step(self):
        observation = self.env.get_observation()
        
        """Temporary fix for Unity as it receives multiple observations"""

        if len(observation.shape) > 1 and self.env.env_name != 'RoboCup': # wonder why this extra condition?
            action = [self.alg.get_action(self.agent, obs, self.env.step_count) for obs in observation]
            next_observation, reward, done, more_data = self.env.step(action)
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                # print(act, rew, don)
                self.buffer.append(exp)
        else:
            action = self.alg.get_action(self.agent, observation, self.env.step_count)
            next_observation, reward, done, more_data = self.env.step(action, discrete_select=self.action_selection_method)
            '''
                @discrete_action argument could be an attribute for the Environment in the config
                Then, we got to be careful which action was actually performed on the environment!
                For example, when @action_selection_method is sample, could be a problem when updating the network
            '''
            t = [observation, action, reward, next_observation, int(done)]
            exp = copy.deepcopy(t)
            self.buffer.append(exp)

        # if self.env.step_count > self.alg.exploration_steps:# and self.env.step_count % 16 == 0:
        #     self.alg.update(self.agent, self.buffer.sample(), self.env.step_count)

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):
        self.env = self.create_environment()
        if self.manual_play:
            from shiva.envs.RoboCupEnvironment import HumanPlayerInterface
            self.HPI = HumanPlayerInterface()
        self.alg = self.create_algorithm()
        if self.load_agents:
            self.agent = Admin._load_agent(self.load_agents)
            self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent(self.get_id())
            if self.using_buffer:
                self.buffer = self.create_buffer()
        print('Launch Successful.')

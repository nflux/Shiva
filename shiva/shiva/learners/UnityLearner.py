from __main__ import shiva
from .Learner import Learner
import helpers.misc as misc
import envs
import algorithms
import buffers
from copy import deepcopy

class UnityLearner(Learner):
    def __init__(self, learner_id, config):
        super(UnityLearner,self).__init__(learner_id, config)
        self.done_counter = 0

    # so now the done is coming from the environment
    def run(self):
        self.step_count = 0
        while not self.env.finished(self.episodes):
            self.exploration_mode = self.step_count < self.alg.exploration_steps
            self.step()
            self.step_count += 1
        self.env.close()

    def step(self):
        observation = self.env.get_observation()
        action = [self.alg.get_action(self.agent, obs, self.step_count) for obs in observation]

        # print("Learner:", observation.shape)
        # print("Learner:", action)

        print('step:', self.step_count, '\treward:', self.env.reward_total)

        next_observation, reward, done, _ = self.env.step(action)
        for obs, act, rew, next_obs, don in zip(observation, action, reward, next_observation, done):
            exp = [obs, act, rew, next_obs, int(don)]
            exp = deepcopy(exp)
            self.buffer.append(exp)
        
        if not self.exploration_mode and self.step_count % 8 == 0:
            self.alg.update(self.agent, self.buffer.sample(), self.step_count)

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = getattr(envs, self.configs['Environment']['type'])
        return environment(self.configs['Environment'])

    def create_algorithm(self):
        algorithm = getattr(algorithms, self.configs['Algorithm']['type'])
        return algorithm(self.env.observation_space, self.env.action_space, [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        
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

        # # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()

        # # Create the agent
        if self.configs['Learner']['load_agents'] is not False:
            self.agent = self.load_agent(self.configs['Learner']['load_agents'])
        else:
            self.agent = self.alg.create_agent(self.get_id())
        
        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]

from settings import shiva
from .Learner import Learner
import helpers.misc as misc
import torch.multiprocessing as mp
import os
import torch
import envs
import algorithms
import buffers
import copy
import random
import numpy as np



class SingleAgentMultiEnvLearner(Learner):
    '''
        Work in progress.
        One MultiEnv Learner for all algorithms

    '''

    def __init__(self, learner_id, config):
        super(SingleAgentMultiEnvLearner,self).__init__(learner_id, config)
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.ep_count = torch.zeros(1).share_memory_()
        self.updates = 1
        self.agent_dir = os.getcwd() + self.agent_path


    def run(self):
        self.step_count = 0
        while self.ep_count < self.episodes:
            while not self.queue.empty():
                exp = self.queue.get()
                observations, actions, rewards, next_observations, dones = zip(*exp)
                print(np.array(rewards).sum())
                for i in range(len(observations)):
                    self.buffer.append([observations[i], actions[i], rewards[i][0], next_observations[i], dones[i][0]])
                self.ep_count += 1
            if self.ep_count.item() / self.configs['Algorithm']['update_episodes'] >= self.updates:
                print(self.ep_count)
                self.alg.update(self.agent,self.old_agent,self.buffer,self.step_count)
                self.agent.save_agent(self.agent_dir,self.step_count)
                self.updates += 1
                print('Copied')
                #Add save policy function here

        # self.p.join()
        print('Hello')
        # del(self.p)
        del(self.queue)


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
        return environment(self.configs['Environment'],self.queue,self.agent,self.ep_count,self.agent_dir,self.episodes)

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
        environment = getattr(envs, self.configs['Environment']['sub_type'])
        self.env = environment(self.configs['Environment'])


        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()
        # Create the agent
        if self.load_agents:
            self.agent= self.load_agent(self.load_agents)
        else:
            self.agent= self.alg.create_agent()
            self.old_agent = self.alg.create_agent()
            self.old_agent = copy.deepcopy(self.agent)


        # Launch the environment
        self.env = self.create_environment()
        # self.p = mp.Process(target = self.env.launch_envs)
        # self.p.start()


        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()



        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]

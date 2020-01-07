# from shiva.core.admin import Admin

from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class
import helpers.misc as misc
import torch.multiprocessing as mp
import os
import torch
import copy
import random
import time
import numpy as np



class SingleAgentMultiEnvLearner(Learner):
    '''
        Work in progress.
        One MultiEnv Learner for all algorithms

    '''

    def __init__(self, learner_id, config):
        super(SingleAgentMultiEnvLearner,self).__init__(learner_id, config)
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.saveLoadFlag = torch.zeros(1).share_memory_()
        self.ep_count = torch.zeros(1).share_memory_()
        self.updates = 5
        self.agent_dir = os.getcwd() + self.agent_path

    def run(self):
        self.step_count = 0
        while self.ep_count < self.episodes:
            while not self.queue.empty():

                exp = self.queue.get()
                if self.configs['Algorithm']['algorithm'] == 'PPO':
                    observations, actions, rewards, logprobs, next_observations, dones = zip(*exp)
                    print("Episode {} Episodic Reward {} ".format(self.ep_count.item(), np.array(rewards).sum()))
                    for i in range(len(observations)):
                        self.step_count += 1
                        self.buffer.append([observations[i], actions[i], rewards[i][0],logprobs[i], next_observations[i], dones[i][0]])
                else:
                    observations, actions, rewards, next_observations, dones = zip(*exp)
                    print("Episode {} Episodic Reward {} ".format(self.ep_count.item(), np.array(rewards).sum()))
                    for i in range(len(observations)):
                        self.buffer.append([observations[i], actions[i], rewards[i][0], next_observations[i], dones[i][0]])
                        self.step_count += 1
                        if self.configs['Algorithm']['algorithm'] != 'PPO':
                            if self.step_count % 64 == 0:
                                self.alg.update(self.agent,self.buffer,self.step_count)
                                self.updates +=1 
                self.ep_count += 1

            if self.ep_count.item() / self.configs['Algorithm']['update_episodes'] >= self.updates:
                
                self.alg.update(self.agent,self.buffer,self.step_count)

                if self.saveLoadFlag.item() == 1:
                    # start_time = time.time()
                    # print("Multi Learner:",self.agent_dir)
                    self.agent.save_agent(self.agent_dir,self.step_count)
                    # print("--- %s seconds ---" % (time.time() - start_time))
                    print("Agent was saved")
                    self.saveLoadFlag[0] = 0

                self.updates += 1
                # print('Copied')
                # Add save policy function here

        # self.p.join()
        #print('Hello')
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
        environment = load_class('shiva.envs', self.configs['Environment']['type'])
        return environment(self.configs['Environment'],self.queue,self.agent,self.ep_count,self.agent_dir,self.episodes, self.saveLoadFlag)

    def create_algorithm(self):
        algorithm = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        acs_continuous = self.env.action_space_continuous
        acs_discrete= self.env.action_space_discrete
        if self.configs['Algorithm']['algorithm'] == 'PPO':
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(), acs_discrete, acs_continuous, [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        else:
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer = load_class('shiva.buffers',self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):
        environment = load_class('shiva.envs', self.configs['Environment']['sub_type'])
        self.env = environment(self.configs)


        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()
        # Create the agent
        if self.load_agents:
            self.agent= self.load_agent(self.load_agents)
        else:
            self.agent= self.alg.create_agent()
            self.agent.save_agent(self.agent_dir,self.step_count)

        # Launch the environment
        self.env = self.create_environment()


        # if buffer set to true in config
        if self.using_buffer:
            # Basic replay buffer at the moment
            self.buffer = self.create_buffer()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]

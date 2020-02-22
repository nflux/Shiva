import torch
import copy
import random
import numpy as np

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class


class IRLLearner(Learner):
    def __init__(self, learner_id, config, port=None):
        super(IRLLearner, self).__init__(learner_id, config, port)
        self.update = False
        self.update_count = 0
        self.launch()

    def launch(self):

        '''

            Here we want to create the three processes

            Maybe it doesn't need to be multiprocessed

            Processes:
            1. First process will be more typical. It will host a PPOAgent with its respective networks.
                - It will interact with the environment to produce trajectories and normalized rewards will be
                assigned using a copy of the rewards network from process three.

            2. Process two will select pairs of segments from the trajectories produced by process 1 & sent to
            either a human or expert to select with of the two pairs is most similar to the desired behavior.

            3. Here the parameters of the rewards network will be updated/optimized using supervised learning.
            Loss will be calculated using cross entropy loss between the predictions of which segments segments
            were selected to be better and the actual human/expert labels.

        '''

        self.env = self.create_environment()
        self.ppo_alg = self.create_algorithm()

        if self.load_agents:
            self.agent = Admin._load_agent(self.load_agents)
            if self.using_buffer:
                self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent(self.get_new_agent_id())
            if self.using_buffer:
                self.buffer = self.create_buffer(self.env.observation_space,
                                                 self.env.action_space['acs_space'] + self.env.action_space['param'])

        self.reward_model = self._create_reward_model(self.env.observation_space, self.env.action_space['acs_space'])

        # load in the expert from path specified in the config
        self.expert = Admin._load_agent(self.expert_path)
        self.irl_alg = self.create_irl_algorithm()
        # I guess the neural network would have to classify the ppo_action as expert or not
        # instantiate a buffer with tuples (state, ppo_action, 0 or 1)
        # 0 indicates it was an expert action where as 1 does not
        # the supervised learning algorithms will predict whether or not the action taken as expert
        self.segment_buffer = self._create_segment_buffer()
        print('Launch Successful.')

        self.run()

    def run(self):
        self.step_count = 0
        # for self.ep_count in range(self.episodes):
        while not self.env.finished(self.episodes):
            self.env.reset()
            # self.totalReward = 0
            # done = False
            # while not done:
            while not self.env.is_done():
                # done = self.step()
                self.step()
                self.step_count += 1
                self.collect_metrics() # metrics per step
            #self.ep_count += 1
            self.collect_metrics(True) # metrics per episode
            #print('Episode {} complete!\tEpisodic reward: {} '.format(self.ep_count, self.env.get_reward_episode()))
            print(self.ep_count)
            if int(self.ep_count / self.configs['Algorithm']['update_episodes']) > self.update_count:
                self.update_count += 1
                #self.ep_count += 1
                self.ppo_alg.update(self.agent, self.buffer, self.step_count)
                self.irl_alg.update(self.irl_agent, self.segment_buffer)
            self.checkpoint()
        del(self.queues)
        self.env.close()

    def step(self):

        observation = self.env.get_observation()

        """Temporary fix for Unity as it receives multiple observations"""
        '''if len(observation.shape) > 1:
            action = [self.old_agent.get_action(obs) for obs in observation]
            next_observation, reward, done, more_data = self.env.step(action)
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                self.buffer.append(exp)'''

        # if len(observation.shape) > 1:
        #     action = [self.agent.get_action(obs) for obs in observation]
        #     logprobs = [self.agent.get_logprobs(obs,act) for obs,act in zip(observation,action)]
        #     next_observation, reward, done, more_data = self.env.step(action)
        #     for i in range(len(action)):
        #         z = copy.deepcopy(zip([observation[i]], [action[i]], [reward[i]], [next_observation[i]], [done[i]], [logprobs[i]]))
        #         for obs, act, rew, next_obs, don, logprob in z:
        #             self.rewards[i] += rew
        #             exp = [obs, act, rew, next_obs, int(don), logprob]
        #             self.queues[i].put(exp)
        #             if don == True:
        #                 print('Episode: ', self.ep_count + 1, ' reward: ', self.rewards[i])
        #                 self.rewards[i] = 0
        #                 self.ep_count += 1
        #                 while not self.queues[i].empty():
        #                     self.buffer.append(self.queues[i].get())
        # else:

        '''
            Currently only supports Gym, if this is successful, it will be extended to RoboCup and Unity
        '''
        action = self.agent.get_action(observation)
        next_observation, _, done, more_data = self.env.step(action)
        reward = self.irl_agent.get_reward(observation, action)
        log_probs = self.agent.get_logprobs(observation, action)
        self.rewards[0] += reward
        t = [observation.numpy(), action, reward, next_observation, int(done), log_probs]
        exp = copy.deepcopy(t)
        if done:
            self.ep_count += 1
            print('Episode: ', self.ep_count, ' reward: ', self.rewards[0])

        self.buffer.append(exp)

    def _create_reward_model(self, obs_space, acs_space):
        reward_model = load_class('shiva.irl', self.configs['IRL']['type'])
        return reward_model(self.configs['URL'])

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs, self.port)

    def create_rl_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),
                               [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_irl_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),
                               [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self, obs_dim, ac_dim):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.env.num_left,
                            obs_dim, ac_dim)

    def _create_segment_buffer(self, obs_space, acs_space):
        buffer_class = load_class('shiva.buffers', self.configs['IRL']['type'])
        return buffer_class(self.configs['IRL']['capacity'], self.configs['IRL']['batch_size'], self.env.num_left,
                            obs_space, acs_space)

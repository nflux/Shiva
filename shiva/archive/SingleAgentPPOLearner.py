import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical
import torch
import envs
import algorithms
import buffers
import copy
import random
import numpy as np

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentPPOLearner(Learner):
    def __init__(self, learner_id, config):
        super(SingleAgentPPOLearner,self).__init__(learner_id, config)
        self.update = False
        self.update_count = 0

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
                self.alg.update(self.agent,self.buffer, self.step_count)
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

        if len(observation.shape) > 1:
            action = [self.agent.get_action(obs) for obs in observation]
            logprobs = [self.agent.get_logprobs(obs,act) for obs,act in zip(observation,action)]
            next_observation, reward, done, more_data = self.env.step(action)
            for i in range(len(action)):
                z = copy.deepcopy(zip([observation[i]], [action[i]], [reward[i]], [next_observation[i]], [done[i]], [logprobs[i]]))
                for obs, act, rew, next_obs, don, logprob in z:
                    self.rewards[i] += rew
                    exp = [obs, act, rew, next_obs, int(don), logprob]
                    self.queues[i].put(exp)
                    if don == True:
                        print('Episode: ', self.ep_count + 1, ' reward: ', self.rewards[i])
                        self.rewards[i] = 0
                        self.ep_count += 1
                        while not self.queues[i].empty():
                            self.buffer.append(self.queues[i].get())
        else:
            action = self.agent.get_action(observation)
            next_observation, reward, done, more_data = self.env.step(action)
            self.rewards[0] += reward
            log_probs = self.agent.get_logprobs(observation,action)
            t = [observation.numpy(), action, reward, next_observation, int(done), log_probs]
            exp = copy.deepcopy(t)
            if done:
                self.ep_count += 1
                #print('Episode: ', self.ep_count, ' reward: ', self.rewards[0])
            self.buffer.append(exp)
        """"""

        # if self.configs['Agent']['action_space'] == 'Discrete':
        #     action= self.old_agent.get_action(observation)
        # elif self.configs['Agent']['action_space'] == 'Continuous':
        #     action = self.old_agent.get_continuous_action(observation)

        # TensorBoard metrics
        # shiva.add_summary_writer(self, self.agent, 'Loss per Step', self.alg.loss, self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Policy Loss per Step', self.alg.policy_loss, self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Value Loss per Step', self.alg.value_loss, self.step_count)
        # shiva.add_summary_writer(self, self.agent, 'Entropy Loss per Step', self.alg.entropy_loss, self.step_count)
        # #shiva.add_summary_writer(self, self.agent, 'Critic Loss per Step', self.alg.get_critic_loss(), self.step_count)
        # #shiva.add_summary_writer(self, self.agent, 'Normalized_Reward_per_Step', reward, self.step_count)
        # #shiva.add_summary_writer(self, self.agent, 'Raw_Reward_per_Step', more_data['raw_reward'], self.step_count)
        # self.totalReward += more_data['raw_reward'][0] if type(more_data['raw_reward']) == list else more_data['raw_reward']
        # t = [observation, action, reward, next_observation, int(done)]
        # deep = copy.deepcopy(t)
        # self.buffer.append(deep)

        # # TensorBoard Metrics
        # if self.done:
        #     # shiva.add_summary_writer(self, self.agent, 'Total Reward per Episode', self.totalReward, self.ep_count)
        #     self.collect_metrics(True) # metrics per episode
        # return done

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),self.configs)

    def create_buffer(self, obs_dim, ac_dim):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if self.env.env_name == 'RoboCupEnvironment':
            return buffer_class(self.configs['Buffer']['capacity'],self.configs['Buffer']['batch_size'], self.env.num_left, obs_dim, ac_dim)
        else:
            return buffer_class(self.configs['Buffer']['capacity'],self.configs['Buffer']['batch_size'], self.env.num_instances, obs_dim, ac_dim)

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):
        self.env = self.create_environment()
        self.queues = [mp.Queue(self.max_length)] * self.env.num_instances
        self.rewards = np.zeros((self.env.num_instances))
        if hasattr(self, 'manual_play') and self.manual_play:
            '''
                Only for RoboCup!
                Maybe for Unity at some point?????
            '''
            from shiva.envs.RoboCupEnvironment import HumanPlayerInterface
            self.HPI = HumanPlayerInterface()
        self.alg = self.create_algorithm()
        if self.load_agents:
            self.agent = Admin._load_agent(self.load_agents)
            if self.using_buffer:
                self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agent = self.alg.create_agent(self.get_new_agent_id())
            if self.using_buffer:
                self.buffer = self.create_buffer(self.env.observation_space, self.env.action_space['acs_space'])

        print('Learners Agents: {}'.format(self.agent))
        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return Admin._load_agents(path)[0]

import torch
import copy
import random
import numpy as np
from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class


class SingleAgentIRLLearner(Learner):

    def __init__(self, learner_id, config, port=None):
        super(SingleAgentIRLLearner, self).__init__(learner_id, config, port)
        self.update = False
        self.step_count = 0
        self.update_count = 0
        self.totalReward = 0
        self.ep_count = 0
        self.episodic_reward = 0
        self.agents = []
        self.launch()

    def launch(self):

        self.env = self._create_environment()
        self.alg = self._create_rl_algorithm()
        self.irl_alg = self._create_irl_algorithm()

        if self.load_agents:
            self.agent = Admin._load_agent(self.load_agents)
            if self.using_buffer:
                self.buffer = Admin._load_buffer(self.load_agents)
        else:
            self.agents.append(self.alg.create_agent())
            self.agents.append(self.irl_alg.create_agent())
            if self.using_buffer:
                self.buffer = self._create_buffer()

        print('Launch Successful.')

        # self.run()

    def run(self):
        # for self.ep_count in range(self.episodes):
        while not self.env.finished(self.episodes):
            self.env.reset()
            self.episodic_reward = 0
            # done = False
            # while not done:
            while not self.env.is_done():
                # done = self.step()
                self.step()
                self.step_count += 1
                # metrics per step
                self.collect_metrics()
            # metrics per episode
            self.collect_metrics(True)
            # print('Episode {} complete!\tEpisodic reward: {} '.format(self.ep_count, self.env.get_reward_episode()))
            print(self.ep_count)
            if int(self.ep_count / self.configs['Algorithm']['update_episodes']) > self.update_count \
                    and self.ep_count > self.configs['Algorithm']['update_episodes']:
                # Use the data in the buffer before PPO clears the buffer
                self.irl_alg.update(self.agents[1], self.buffer, self.step_count)
                self.alg.update(self.agents[0], self.buffer, self.step_count)
                self.update_count += 1
            # Burn in so the reward estimator gives rewards with some merit behind them
            elif int(self.ep_count / self.configs['Algorithm']['update_episodes']) > self.update_count:
                self.irl_alg.update(self.agents[1], self.buffer, self.step_count)
            self.checkpoint()
        # del self.queues
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
        action = self.agents[0].get_action(observation)
        next_observation, _, done, more_data = self.env.step(action)
        log_probs = self.agents[0].get_logprobs(observation, action)
        self.reward = self.agents[1].get_reward(torch.tensor(observation).float(), torch.tensor(action).float())
        self.episodic_reward += self.reward
        exp = copy.deepcopy([
                             torch.tensor(observation),
                             torch.tensor(action),
                             torch.tensor(self.reward).view(-1, 1),
                             torch.tensor(next_observation),
                             torch.tensor(int(done)),
                             torch.tensor(log_probs)
                            ])
        if done:
            print(f'Episode:  {self.ep_count}  Episodic Reward: {self.episodic_reward} Step Count: {self.env.steps_per_episode}')
            self.ep_count += 1
        self.buffer.push(exp)

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [

                ('Stepwise_Reward', self.reward),
                ('IRL_Algorithm/Loss_per_Step', self.irl_alg.loss),
                ('Algorithm/Loss_per_Step', self.alg.loss),
                ('Algorithm/Policy_Loss_per_Step', self.alg.policy_loss),
                ('Algorithm/Value_Loss_per_Step', self.alg.value_loss),
                ('Algorithm/Entropy_Loss_per_Step', self.alg.entropy_loss),
            ]
        else:
            metrics = [
                ('Episodic_Reward', self.episodic_reward)
            ]
        return metrics

    def _create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs, self.port)

    def _create_rl_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),
                               [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def _create_irl_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['IRLAlgorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(),
                               [self.configs['IRLAlgorithm'], self.configs['IRLAgent'], self.configs['IRLNetwork']])

    def _create_buffer(self):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'], 1,
                            self.env.get_observation_space(), self.env.get_action_space()['acs_space'])


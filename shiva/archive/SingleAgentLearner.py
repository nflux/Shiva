import torch
import copy
import random
import numpy as np

from shiva.core.admin import Admin
from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class

class SingleAgentLearner(Learner):
    def __init__(self, learner_id, config, port=None):
        super(SingleAgentLearner ,self).__init__(learner_id, config, port)

    def run(self):
        if self.evaluate:
            print("** Running Evaluation mode **")
        self.step_count_per_run = 0
        while not self.env.finished(self.episodes):
            self.env.reset()
            while not self.env.is_done():
                self.step()

                if not self.evaluate and not (self.configs['Algorithm']['algorithm'] == 'PPO'):
                    self.alg.update(self.agent, self.buffer, self.env.step_count)
                self.collect_metrics()
                if self.is_multi_process_cutoff(): return None # PBT Cutoff
                else: continue
            if not self.evaluate:
                if self.configs['Algorithm']['algorithm'] == 'PPO':
                    if self.buffer.size >= self.configs['Buffer']['batch_size']:
                        self.alg.update(self.agent, self.buffer, self.env.step_count, episodic=True)
                else:
                    self.alg.update(self.agent, self.buffer, self.env.step_count, episodic=True)
            # this is one hundred percent an episodic agent noise reset
            if self.agent.__str__() == 'DDPGAgent':
                self.agent.ou_noise.reset()
            self.collect_metrics(episodic=True)
            self.checkpoint()
            print('Step # {}\tEpisode {} completed on {} steps!\tEpisodic reward: {} '.format(self.env.step_count, self.env.done_count, self.env.steps_per_episode, self.env.reward_per_episode))
        self.env.close()

    def step(self):
        observation = self.env.get_observation()

        """Temporary fix for Unity as it receives multiple observations"""

        if self.env.env_name == 'RoboCup':

            action = self.agent.get_action(observation, self.env.step_count, self.evaluate)

            next_observation, reward, done, more_data = self.env.step(action, discrete_select=self.action_selection_method,device=self.device)

            exp = list(map(torch.clone, (torch.from_numpy(observation), action, torch.from_numpy(reward),
                                                torch.from_numpy(next_observation), torch.from_numpy(np.array([done])).bool()) ))

        elif self.configs['Algorithm']['algorithm'] == 'PPO':
            if len(observation.shape) > 1:
                action = self.agent.get_action(observation)
                logprobs = self.agent.get_logprobs(observation,action).sum(-1,keepdim=True)
                next_observation, reward, done, more_data = self.env.step(action)

                exp = copy.deepcopy([
                            torch.tensor(observation),
                            torch.tensor(action[0]),
                            torch.tensor(reward).reshape(-1,1),
                            torch.tensor(next_observation),
                            torch.tensor(done).reshape(-1,1),
                            logprobs.clone().detach().requires_grad_(True)
                    ])

            else:
                action = self.agent.get_action(observation)
                logprobs = self.agent.get_logprobs(observation,action).sum(-1,keepdim=True)
                next_observation, reward, done, more_data = self.env.step(action, self.action_selection_method)

                exp = copy.deepcopy([
                            torch.tensor(observation),
                            torch.tensor(action),
                            torch.tensor(reward).reshape(-1,1),
                            torch.tensor(next_observation),
                            torch.tensor(done).reshape(-1,1),
                            logprobs.clone().detach().requires_grad_(True)
                    ])
            self.buffer.push(exp)

        else:
            if len(observation.shape) > 1:
                action = [self.agent.get_action(obs, self.env.step_count) for obs in observation]
                next_observation, reward, done, more_data = self.env.step(action)
                # print(action)
                exp = copy.deepcopy([
                            torch.tensor(observation),
                            torch.tensor(action[0]),
                            torch.tensor(reward).reshape(-1,1),
                            torch.tensor(next_observation),
                            torch.tensor(done).reshape(-1,1)
                    ])
            else:
                action = self.agent.get_action(observation, self.env.step_count)
                next_observation, reward, done, more_data = self.env.step(action, self.action_selection_method)
                exp = copy.deepcopy([
                            torch.tensor(observation),
                            torch.tensor(action),
                            torch.tensor(reward).reshape(-1,1),
                            torch.tensor(next_observation),
                            torch.tensor(done).reshape(-1,1)
                    ])

            self.buffer.push(exp)

        """"""


    def is_multi_process_cutoff(self):
        ''' FOR MULTIPROCESS PBT PURPOSES '''
        self.step_count = self.env.step_count
        self.ep_count = self.env.done_count
        try:
            if self.multi and self.step_count_per_run >= self.updates_per_iteration:
                return True
        except:
            pass
        self.step_count_per_run += 1
        return False

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.env.get_observation_space(), self.env.get_action_space(), self.configs)

    def create_buffer(self, obs_dim, ac_dim):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if self.env.env_name == 'RoboCupEnvironment':
            return buffer_class(self.configs['Buffer']['capacity'],self.configs['Buffer']['batch_size'], self.env.num_left, obs_dim, ac_dim)
        else:
            return buffer_class(self.configs['Buffer']['capacity'],self.configs['Buffer']['batch_size'], self.env.num_instances, obs_dim, ac_dim)


    def launch(self):
        self.env = self.create_environment()
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
        print('Launch Successful.')

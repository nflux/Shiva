import os
import numpy as np
import time

from mlagents.envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot

class UnityWrapperEnvironment(Environment):
    def __init__(self, config):
        assert UnityEnvironment.API_VERSION == 'API-11' or UnityEnvironment.API_VERSION == 'API-12', 'Shiva only support mlagents api v11 or v12'

        super(UnityWrapperEnvironment, self).__init__(config)
        self.worker_id = 0
        self._connect()
        # self.debug()
        self.action_space = self.BrainParameters.vector_action_space_size[0]
        self.observation_space = self.BrainParameters.vector_observation_space_size * self.BrainParameters.num_stacked_vector_observations
        
        self.action_space_discrete = self.action_space if self.BrainParameters.vector_action_space_type == 'discrete' else None
        self.action_space_continuous = self.action_space if self.BrainParameters.vector_action_space_type == 'continuous' else None

        self.num_instances = len(self.BrainInfo.agents)
        
        self.rewards = np.array(self.BrainInfo.rewards)
        self.reward_total = 0
        self.dones = np.array(self.BrainInfo.local_done)
        
        self.step_count = 0
        self.load_viewer()

    def _connect(self):
        self.brain_name = self.env_name
        self.channel = EngineConfigurationChannel()
        self.Unity = UnityEnvironment(
            file_name= self.exec,
            base_port = self.port,
            seed = self.configs['Algorithm']['manual_seed'],
            side_channels = [self.channel],
            no_graphics= not self.render
        )
        self._reset()

    # def _connect(self):
    #     self._call()
    #     try:
    #         self._call()
    #         print('Worker id:', self.worker_id)
    #     except:
    #         if self.worker_id < 3:
    #             # try to connect with different worker ids
    #             self.worker_id += 1
    #             # self._connect()
    #         else:
    #             assert False, 'Enough worker_id tries.'

    def _reset(self, new_config=None):
        '''
            This only gets called once at connection with Unity server
            UnityEnvironment v11
        '''
        if new_config is not None:
            self.reset_params = new_config
        self.BrainInfoDict = self.Unity.reset(train_mode=self.train_mode, config=self.reset_params)
        self.BrainInfo = self.BrainInfoDict[self.brain_name]
        self.BrainParameters = self.Unity.brains[self.brain_name]
        self.observations = self.BrainInfo.vector_observations
        self.rewards = self.BrainInfo.rewards
        self.dones = self.BrainInfo.local_done
        self.reset()

    # def _reset(self, new_config=None):
    #     '''
    #         This only gets called once at connection with Unity server
    #         UnityEnvironment v12.1
    #     '''
    #     if new_config is not None:
    #         self.reset_params = new_config
    #     self.Unity.reset()
    #     self.BrainInfo = self.BrainInfoDict[self.brain_name]
    #     self.BrainParameters = self.Unity.brains[self.brain_name]
    #     self.observations = self.BrainInfo.vector_observations
    #     self.rewards = self.BrainInfo.rewards
    #     self.dones = self.BrainInfo.local_done
    #     self.reset()

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize the temporary done counter due to the multiagents on Unity
        '''
        self.steps_per_episode = 0
        self.temp_done_counter = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0

    def get_random_action(self):
        return np.array([np.random.uniform(-1, 1, size=self.action_space) for _ in range(self.num_instances)])

    def step(self, actions):
        self.actions = self._clean_actions(actions)
        self.BrainInfoDict = self.Unity.step(self.actions)
        self.BrainInfo = self.BrainInfoDict[self.brain_name]
        self.observations = np.array(self.BrainInfo.vector_observations)
        self.rewards = np.array(self.BrainInfo.rewards)
        self.dones = np.array(self.BrainInfo.local_done)
        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode --> is equal to the amount of instances on Unity, 1 Shiva step could be a couple of Unity steps
                Cumulative # of steps           self.step_count
                Temporary episode count         self.temp_done_counter --> used for the is_done() call. Zeroed on reset().
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total
        '''
        self.steps_per_episode += self.num_instances
        self.step_count += self.num_instances
        self.temp_done_counter += sum([ 1 if val else 0 for val in self.dones ])
        self.done_count += self.temp_done_counter
        self.reward_per_step = sum(self.rewards) / self.num_instances
        self.reward_per_episode += self.reward_per_step
        self.reward_total += self.reward_per_episode

        # self.debug()
        return self.observations, self.rewards, self.dones, {}

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step)
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return metrics

    def is_done(self):
        '''
            One Shiva episode will be playing the number of instances in Unity
        '''
        return self.temp_done_counter >= self.num_instances

    def _clean_actions(self, actions):
        '''
            Get the argmax when the Action Space is Discrete
        '''
        if self.BrainParameters.vector_action_space_type == 'discrete':
            actions = np.array([np.argmax(_act) for _act in actions])
        return actions

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_observation(self):
        return self.observations

    def get_actions(self):
        return self.actions

    def get_reward(self):
        return sum(self.rewards)/self.num_instances

    def get_total_reward(self):
        return self.reward_per_episode

    def load_viewer(self):
        pass

    def close(self):
        self.Unity.close()
        delattr(self, 'Unity')

    def debug(self):
        print('self -->', self.__dict__)
        print('BrainInfo -->', self.BrainInfo.__dict__)
        print('BrainParameters -->', self.BrainParameters.__dict__)

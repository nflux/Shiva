from .Environment import Environment
from mlagents.envs.environment import UnityEnvironment
import os
import numpy as np
import time

from helpers.misc import action2one_hot

class UnityWrapperEnvironment(Environment):
    def __init__(self, config):
        assert UnityEnvironment.API_VERSION == 'API-11', 'Shiva only support mlagents api v11'

        super(UnityWrapperEnvironment, self).__init__(config)
        self.worker_id = 0
        self._connect()

        self.action_space = self.BrainParameters.vector_action_space_size[0]
        self.observation_space = self.BrainParameters.vector_observation_space_size * self.BrainParameters.num_stacked_vector_observations
        
        self.action_space_discrete = self.action_space if self.BrainParameters.vector_action_space_type == 'discrete' else None
        self.action_space_continuous = self.action_space if self.BrainParameters.vector_action_space_type == 'continuous' else None

        self.num_instances = len(self.BrainInfo.agents)
        
        self.rewards = np.array(self.BrainInfo.rewards)
        self.reward_total = 0
        self.dones = np.array(self.BrainInfo.local_done)
        self.done_count = 0
        self.temp_done_counter = 0 # since we have @self.num_instances, a partial done will be when all instances finished at least once
        
        self.step_count = 0
        self.load_viewer()

    def _call(self):
        self.brain_name = self.env_name
        self.Unity = UnityEnvironment(file_name=self.exec, worker_id=self.worker_id, no_graphics= not self.render)
        self._reset()

    def _connect(self):
        try:
            self._call()
            print('Worker id:', self.worker_id)
        except:
            if self.worker_id < 200:
                # try to connect with different worker ids
                self.worker_id += 1
                self._connect()
            else:
                assert False, 'Enough worker_id tries.'

    def _reset(self, new_config=None):
        '''
            This only gets called once at connection with Unity server
        '''
        if new_config is not None:
            self.reset_params = new_config
        self.BrainInfoDict = self.Unity.reset(train_mode=self.train_mode, config=self.reset_params)
        self.BrainInfo = self.BrainInfoDict[self.brain_name]
        self.BrainParameters = self.Unity.brains[self.brain_name]
        self.observations = self.BrainInfo.vector_observations
        self.rewards = self.BrainInfo.rewards
        self.dones = self.BrainInfo.local_done

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize the temporary done counter
        '''
        self.temp_done_counter = 0

    def get_random_action(self):
        return np.array([np.random.uniform(-1, 1, size=self.action_space) for _ in range(self.num_instances)])

    def step(self, actions):
        """
            Need to do some manipulation of the data when many instances map to 1 agent
        """
        # make sure discrete side is one hot encoded
        self.actions = self._clean_actions(actions)
        # self.actions = actions
        self.BrainInfoDict = self.Unity.step(self.actions)
        self.BrainInfo = self.BrainInfoDict[self.brain_name]
        self.observations = np.array(self.BrainInfo.vector_observations)
        self.rewards = np.array(self.BrainInfo.rewards)
        self.dones = np.array(self.BrainInfo.local_done)
        
        self.step_count += 1 # this are Shiva steps
        self.done_count += sum([ 1 if val else 0 for val in self.dones ])
        self.temp_done_counter += self.done_count
        self.reward_total += sum(self.rewards) / self.num_instances

        # self.debug()
        return self.observations, self.rewards, self.dones, {}

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [('Raw_Reward_per_Step', sum(self.rewards)/self.num_instances)]
        else:
            metrics = [('Raw_Rewards_per_Episode', self.reward_total)]
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

    def get_observation(self):
        return self.observations

    def get_actions(self):
        return self.actions

    def get_reward(self):
        return sum(self.rewards)/self.num_instances

    def load_viewer(self):
        pass

    def close(self):
        self.Unity.close()
        delattr(self, 'Unity')

    def debug(self):
        print('self -->', self.__dict__)
        print('BrainInfo -->', self.BrainInfo.__dict__)
        print('BrainParameters -->', self.BrainParameters.__dict__)

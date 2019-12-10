from .Environment import Environment
from mlagents.envs.environment import UnityEnvironment
import os
import numpy as np

from helpers.misc import action2one_hot

class UnityWrapperEnvironment(Environment):
    def __init__(self, config):
        super(UnityWrapperEnvironment, self).__init__(config)
        self.worker_id = 0
        self._connect()

        self.action_space = self.BrainParameters.vector_action_space_size[0]
        self.observation_space = self.BrainParameters.vector_observation_space_size * self.BrainParameters.num_stacked_vector_observations
        
        # self.action_discrete_size = self.discrete_action_size
        # self.action_continuous_size = self.action_space - self.action_discrete_size

        self.num_instances = len(self.BrainInfo.agents)
        
        self.rewards = np.array(self.BrainInfo.rewards)
        self.reward_total = 0
        self.dones = np.array(self.BrainInfo.local_done)
        self.done_counts = 0
        
        self.step_count = 0
        self.load_viewer()

    def _call(self):
        self.brain_name = self.env_name
        self.Unity = UnityEnvironment(file_name=self.exec, worker_id=self.worker_id, no_graphics=self.render)
        self.reset()

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

    def reset(self, new_config=None):
        if new_config is not None:
            self.reset_params = new_config
        self.BrainInfoDict = self.Unity.reset(train_mode=self.train_mode, config=self.reset_params)
        self.BrainInfo = self.BrainInfoDict[self.brain_name]
        self.BrainParameters = self.Unity.brains[self.brain_name]
        self.observations = self.BrainInfo.vector_observations
        self.rewards = self.BrainInfo.rewards
        self.dones = self.BrainInfo.local_done

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
        self.done_counts += sum([ 1 if val else 0 for val in self.dones ])
        self.reward_total += sum(self.rewards) / self.num_instances

        # self.debug()

        # return self._map_data(self.observations, self.rewards, self.dones, {})
        return self.observations, self.rewards, self.dones, {}

    def _map_data(self, observations, rewards, dones, extra_data):
        """
            Maps data to the amount of Shiva Learners
            Is required to chop the data for all the @self.num_instances
        """
        # print(observations.shape, rewards.shape, dones.shape)
        return observations.flatten(), rewards.flatten(), dones.flatten(), extra_data

    def _clean_actions(self, actions):
        # print('before:',actions)
        if self.BrainParameters.vector_action_space_type == 'discrete':
            actions = np.array([np.argmax(_act) for _act in actions])
        # print('after:',actions)
        # input()
        return actions

    def get_observation(self):
        return self.observations

    def get_actions(self):
        return self.actions

    def get_reward(self):
        return self.rewards

    def load_viewer(self):
        pass

    def close(self):
        self.Unity.close()
        delattr(self, 'Unity')

    def debug(self):
        print('self -->', self.__dict__)
        print('BrainInfo -->', self.BrainInfo.__dict__)
        print('BrainParameters -->', self.BrainParameters.__dict__)

from .robocup.rc_env import rc_env
from .Environment import Environment
from mlagents.envs.environment import UnityEnvironment
import os
import numpy as np

from helpers.misc import action2one_hot

class UnityEnvironment2(Environment):
    def __init__(self, config):
        super(UnityEnvironment2, self).__init__(config)
        try:
            self.connect()
        except:
            assert False, "Mmm.. try changing UnityEnvironment worker_id"
            self.close()

        self.action_space_size = self.BrainParameters.vector_action_space_size[0]
        self.observ_space_size = self.BrainParameters.vector_observation_space_size
        self.action_discrete_size = self.discrete_action_size
        self.action_continuous_size = self.action_space_size - self.action_discrete_size

        self.num_instances = len(self.BrainInfo.agents)
        
        self.rewards = self.BrainInfo.rewards
        self.reward_total = 0
        self.dones = self.BrainInfo.local_done
        self.done_counts = 0
        
        self.step_count = 0
        self.load_viewer()

        self.debug()

    def connect(self):
        self.brain_name = self.env_name
        self.Unity = UnityEnvironment(file_name=self.exec, worker_id=self.worker_id)
        self._reset()

    def _reset(self):
        self.BrainInfo = self.Unity.reset(train_mode=self.train_mode)[self.brain_name]
        self.BrainParameters = self.Unity.brains[self.brain_name]
        self.observations = self.BrainInfo.vector_observations
        self.rewards = self.BrainInfo.rewards
        self.dones = self.BrainInfo.local_done

    def get_random_action(self):
        return np.array([np.random.uniform(-1, 1, size=self.action_space_size) for _ in range(self.num_instances)])

    def step(self, actions):
        """
            Need to do some manipulation of the data when many instances map to 1 agent
        """
        # make sure discrete side is one hot encoded
        self.actions = self._clean_actions(actions)
        # self.actions = actions

        self.BrainInfo = self.Unity.step(self.actions)[self.brain_name]
        self.observations = self.BrainInfo.vector_observations
        self.rewards = self.BrainInfo.rewards
        self.dones = self.BrainInfo.local_done
        self.done_counts += sum([ 1 if val else 0 for val in self.dones ])
        self.reward_total += sum(self.rewards) / self.num_instances

        self.debug()
        return self.observations, self.rewards, self.dones, {}

    def _clean_actions(self, actions):
        # print('before:',actions)
        if self.BrainParameters.vector_action_space_type == 'discrete':
            actions = np.argmax(actions)
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

    def debug(self):
        print('self -->', self.__dict__)
        print('BrainInfo -->', self.BrainInfo.__dict__)
        print('BrainParameters -->', self.BrainParameters.__dict__)

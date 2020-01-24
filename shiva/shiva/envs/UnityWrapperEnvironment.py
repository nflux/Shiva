import os
import numpy as np
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot

class UnityWrapperEnvironment(Environment):
    def __init__(self, config):
        assert UnityEnvironment.API_VERSION == 'API-12', 'Shiva only support mlagents v12'
        self.on_policy = False
        super(UnityWrapperEnvironment, self).__init__(config)
        self.worker_id = config['worker_id'] if 'worker_id' in config else 0
        self._connect()
        self.set_initial_values()

    def _connect(self):
        self.group_id = self.env_name
        self.channel = EngineConfigurationChannel()
        self.Unity = UnityEnvironment(
            file_name= self.exec,
            base_port = self.port if hasattr(self, 'port') else 5005, # 5005 is Unity's default value
            worker_id = self.worker_id,
            seed = self.configs['Algorithm']['manual_seed'],
            side_channels = [self.channel],
            no_graphics= not self.render
        )
        self.Unity.reset()
        self.groups = self.Unity.get_agent_groups()

        '''Assuming to control 1 Group ID on Unity here'''
        assert self.group_id in self.groups, "Wrong env_name provided.. (corresponds to Unity's Agent Group ID)"
        self.GroupSpec = self.Unity.get_agent_group_spec(self.group_id)
        self.reset()

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics. Unity resets the environment on its own..
        '''
        self.steps_per_episode = 0
        self.temp_done_counter = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0

    def set_initial_values(self):
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.batched_step_results = self.Unity.get_step_result(self.group_id)

        # same behaviour
        self.num_instances = self.batched_step_results.n_agents() # this is unique for Unity as it creates agents with same behaviours
        self.instances_ids = self.batched_step_results.agent_id # unique for Unity

        # diff behaviour
        self.num_agents = 1 # agents with different behaviour
        self.agents_id = [0] # diff behaviour agents ids
        # self.debug()
        self.observations = self.batched_step_results.obs[0]
        self.rewards = self.batched_step_results.reward
        self.dones = self.batched_step_results.done

        self.reward_total = 0
        self.step_count = 0

    def get_random_action(self):
        return np.array([np.random.uniform(-1, 1, size=self.action_space) for _ in range(self.num_instances)])

    def step(self, actions):
        self.actions = self._clean_actions(actions)
        self.Unity.set_actions(self.group_id, self.actions)
        self.Unity.step()
        self.batched_step_results = self.Unity.get_step_result(self.group_id)

        self.observations = self.batched_step_results.obs[0]
        self.rewards = self.batched_step_results.reward
        self.dones = self.batched_step_results.done
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
        self.done_count += sum([ 1 if val else 0 for val in self.dones ])
        self.reward_per_step = sum(self.rewards) / self.num_instances
        self.reward_per_episode += self.reward_per_step
        self.reward_total += self.reward_per_episode

        # self.debug()
        return self.observations, self.rewards, self.dones, {}

    def get_metrics(self, episodic=True):
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
            One Shiva episode is when all instances in the Environment terminate at least once
        '''
        if not self.on_policy:
            return self.temp_done_counter >= self.num_instances
        else:
            return self.temp_done_counter >= self.update_episodes

    def _clean_actions(self, actions):
        '''
            Get the argmax when the Action Space is Discrete
            else,
                make sure it's numpy array
        '''
        if self.GroupSpec.is_action_discrete():
            actions = np.array([[np.argmax(_act)] for _act in actions])
        elif type(actions) != np.ndarray:
            actions = np.array(actions)
        return actions

    def get_action_space(self):
        if self.GroupSpec.is_action_discrete():
            self.num_branches = self.GroupSpec.action_size # this is the number of independent actions
            # we currently only deal with 1 independent action
            # return self.GroupSpec.discrete_action_branches[0] # grab the first branch only
            return {
                'discrete': self.GroupSpec.discrete_action_branches[0],
                'param': 0,
                'acs_space': self.GroupSpec.discrete_action_branches[0]
            }
        elif self.GroupSpec.is_action_continuous():
            # return self.GroupSpec.action_size
            return {
                'discrete': 0,
                'param': self.GroupSpec.action_size,
                'acs_space': self.GroupSpec.action_size
            }

    def get_observation_space(self):
        '''
            Unsure why Unity does the double indexing on observations..
        '''
        return self.GroupSpec.observation_shapes[0][0]

    def get_observations(self):
        return self.observations

    def get_observation(self):
        return self.observations

    def get_actions(self):
        return self.actions
    def get_action(self):
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
        try:
            print('self -->', self.__dict__)
            # print('UnityEnv -->', self.Unity.__dict__)
            print('GroupSpec -->', self.GroupSpec)
            print('BatchStepResults -->', self.batched_step_results.__dict__)
        except:
            print('debug except')

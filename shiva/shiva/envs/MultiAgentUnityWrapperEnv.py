import os
import numpy as np
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

from shiva.envs.Environment import Environment
from shiva.helpers.misc import action2one_hot

class MultiAgentUnityWrapperEnv(Environment):
    def __init__(self, config):
        assert UnityEnvironment.API_VERSION == 'API-12', 'Shiva only support mlagents v12'
        self.on_policy = False
        super(MultiAgentUnityWrapperEnv, self).__init__(config)
        self.worker_id = config['worker_id'] if 'worker_id' in config else 0
        self._connect()
        self.set_initial_values()

    def _connect(self):
        self.channel = {
            'config': EngineConfigurationChannel(),
            'props': FloatPropertiesChannel()
        }
        self.Unity = UnityEnvironment(
            file_name= self.exec,
            base_port = self.port if hasattr(self, 'port') else 5005, # 5005 is Unity's default value
            worker_id = self.worker_id,
            seed = self.worker_id * 5005,
            side_channels = [self.channel['config'], self.channel['props']],
            no_graphics= not self.render,
            timeout_wait = self.timeout_wait if hasattr(self, 'timeout_wait') else 60
        )
        # self.channel['config'].set_configuration_parameters(**self.unity_configs)
        # for k, v in self.unity_props.items():
        #     self.channel['props'].set_property(k, v)

        self.Unity.reset()

    def set_initial_values(self):
        '''Unique Agent Behaviours'''
        self.roles = self.Unity.get_agent_groups()
        self.num_agents = len(self.roles)

        '''Grab all the Agents Specs'''
        self.GroupSpec = {role:self.Unity.get_agent_group_spec(role) for role in self.roles}

        self.action_space = {role:self.get_action_space_from_unity_spec(self.GroupSpec[role]) for role in self.roles}
        self.observation_space = {role:self.get_observation_space_from_unity_spec(self.GroupSpec[role]) for role in self.roles}

        self.collect_step_data()

        '''Calculate how many instances Unity has'''
        self.num_instances_per_role = {role:self.BatchedStepResult[role].n_agents() for role in self.roles}
        self.num_instances_per_env = int( sum(list(self.num_instances_per_role.values())) / self.num_agents ) # all that matters is that is > 1

        '''Init session cumulative metrics'''
        self.reward_total = {role:0 for role in self.roles}
        self.step_count = 0
        self.done_count = 0
        '''Reset Metrics'''
        self.reset()

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics. Unity resets the environment on its own.
        '''
        self.steps_per_episode = 0
        self.temp_done_counter = 0
        self.reward_per_step = {role:0 for role in self.roles}
        self.reward_per_episode = {role:0 for role in self.roles}

    def collect_step_data(self):
        self.BatchedStepResult = {role:self.Unity.get_step_result(role) for role in self.roles}
        self.observations = {role:self._flatten_observations(self.BatchedStepResult[role].obs) for role in self.roles}
        self.rewards = {role:self.BatchedStepResult[role].reward for role in self.roles}
        self.dones = {role:self.BatchedStepResult[role].done for role in self.roles}

    def _flatten_observations(self, obs):
        '''Turns the funky (2, 16, 56) array into a (16, 112)'''
        return np.concatenate([o for o in obs], axis=-1)

    def step(self, actions):
        self.actions = {role:self._clean_actions(role, actions[ix]) for ix, role in enumerate(self.roles)}
        { self.Unity.set_actions(role, self.actions[role]) for role in self.roles }

        self.Unity.step()
        self.collect_step_data()
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
        self.steps_per_episode += self.num_instances_per_env
        self.step_count += self.num_instances_per_env
        self.temp_done_counter += sum(sum(self.dones[role]) for role in self.roles)
        self.done_count += sum(sum(self.dones[role]) for role in self.roles)
        for role in self.roles:
            # in case there's asymetric environment
            self.reward_per_step[role] += sum(self.BatchedStepResult[role].reward) / self.BatchedStepResult[role].n_agents()
            self.reward_per_episode[role] += self.reward_per_step[role]
            self.reward_total[role] += self.reward_per_episode[role]

        return list(self.observations.values()), list(self.rewards.values()), list(self.dones.values()), {}

    def get_metrics(self, episodic=True):
        '''MultiAgent Metrics'''
        metrics = {role:self.get_role_metrics(role, episodic) for ix, role in enumerate(self.roles)}
        return list(metrics.values())

    def get_role_metrics(self, role=None, episodic=True):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step[role])
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode[role]),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]
        return metrics

    def is_done(self, n_episodes=None):
        '''
            One Shiva episode is when all instances in the Environment terminate at least once
            On MultiAgent env, all agents will have the same number of dones, so we can check only one of them
        '''
        # return self.temp_done_counter > 0
        # print("{} {}".format(self.temp_done_counter, self.num_instances_per_env))
        return self.temp_done_counter >= self.num_instances_per_env

    def _clean_actions(self, role, role_actions):
        '''
            Get the argmax when the Action Space is Discrete
            else,
                make sure it's numpy array
        '''
        # assert group_actions.shape == (self.BatchedStepResult[group].n_agents(), 1), "Actions for group {} should be of shape {}".format(group, (self.BatchedStepResult[group].n_agents(), 1))
        if self.GroupSpec[role].is_action_discrete():
            actions = np.array([ [np.argmax(_act)] for _act in role_actions ])
        elif type(role_actions) != np.ndarray:
            actions = np.array(role_actions)
        return actions

    def get_action_space_from_unity_spec(self, unity_spec):
        if unity_spec.is_action_discrete():
            return {
                'discrete': unity_spec.action_shape[0],
                'continuous': 0,
                'param': 0,
                'acs_space': unity_spec.action_shape[0]
            }
        elif unity_spec.is_action_continuous():
            return {
                'discrete': 0,
                'continuous': unity_spec.action_size,
                'param': unity_spec.action_size,
                'acs_space': unity_spec.action_size
            }
        else:
            assert "Something weird happened here..."

    def get_observation_space_from_unity_spec(self, unity_spec):
        # flatten the obs_shape, g.e. from [(56,), (56,)] to 112
        return sum([ sum(obs_shape) for obs_shape in unity_spec.observation_shapes])

    def get_observations(self):
        return list(self.observations.values())

    def get_actions(self):
        return list(self.actions.values())

    def get_rewards(self):
        return list(self.rewards.values())

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

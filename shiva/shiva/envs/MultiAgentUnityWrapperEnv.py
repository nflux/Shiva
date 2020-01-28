import os
import numpy as np
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

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

    def set_initial_values(self):
        '''Unique Agent Behaviours'''
        self.agent_groups = self.Unity.get_agent_groups()
        self.num_agents = len(self.agent_groups)

        '''Grab all the Agents Specs'''
        self.GroupSpec = {group:self.Unity.get_agent_group_spec(group) for group in self.agent_groups}

        self.action_space = {group:self.get_action_space_from_unity_spec(self.GroupSpec[group]) for group in self.agent_groups}
        self.observation_space = {group:self.get_observation_space_from_unity_spec(self.GroupSpec[group]) for group in self.agent_groups}

        self.collect_step_data()

        '''Calculate how many instances Unity has'''
        self.num_instances_per_group = {group:self.BatchedStepResult[group].n_agents() for group in self.agent_groups}
        self.num_instances_per_env = int( sum(list(self.num_instances_per_group.values())) / self.num_agents ) # all that matters is that is > 1

        '''Init session cumulative metrics'''
        self.reward_total = {group:0 for group in self.agent_groups}
        self.step_count = 0 #{group:0 for group in self.agent_groups}
        self.done_count = 0 #{group:0 for group in self.agent_groups}
        '''Reset Metrics'''
        self.reset()

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics. Unity resets the environment on its own.
        '''
        self.steps_per_episode = 0 #{group:0 for group in self.agent_groups}
        self.temp_done_counter = 0 #{group:0 for group in self.agent_groups}
        self.reward_per_step = {group:0 for group in self.agent_groups}
        self.reward_per_episode = {group:0 for group in self.agent_groups}

    def collect_step_data(self):
        self.BatchedStepResult = {group:self.Unity.get_step_result(group) for group in self.agent_groups}
        self.observations = {group:self._flatten_observations(self.BatchedStepResult[group].obs) for group in self.agent_groups}
        self.rewards = {group:self.BatchedStepResult[group].reward for group in self.agent_groups}
        self.dones = {group:self.BatchedStepResult[group].done for group in self.agent_groups}

    def _flatten_observations(self, obs):
        '''Turns the funky (2, 16, 56) array into a (16, 112)'''
        return np.concatenate([o for o in obs], axis=-1)

    def step(self, actions):
        self.actions = {group:self._clean_actions(group, actions[ix]) for ix, group in enumerate(self.agent_groups)}
        { self.Unity.set_actions(group, self.actions[group]) for group in self.agent_groups }

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
        self.temp_done_counter += sum(sum(self.dones[group]) for group in self.agent_groups)
        self.done_count += sum(sum(self.dones[group]) for group in self.agent_groups)
        for group in self.agent_groups:
            # in case there's asymetric environment
            self.reward_per_step[group] += sum(self.BatchedStepResult[group].reward) / self.BatchedStepResult[group].n_agents()
            self.reward_per_episode[group] += self.reward_per_step[group]
            self.reward_total[group] += self.reward_per_episode[group]

            # self.steps_per_episode[group] += self.BatchedStepResult[group].n_agents()
            # self.step_count[group] += self.BatchedStepResult[group].n_agents()
            # self.temp_done_counter[group] += sum(self.dones[group])
            # self.done_count[group] += sum(self.dones[group])
            # self.reward_per_step[group] += sum(self.BatchedStepResult[group].reward) / self.BatchedStepResult[group].n_agents()
            # self.reward_per_episode[group] += self.reward_per_step[group]
            # self.reward_total[group] += self.reward_per_episode[group]

        # self.debug()
        return list(self.observations.values()), list(self.rewards.values()), list(self.dones.values()), {}

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

    def is_done(self, n_episodes=None):
        '''
            One Shiva episode is when all instances in the Environment terminate at least once
            On MultiAgent env, all agents will have the same number of dones, so we can check only one of them
        '''
        return self.temp_done_counter > 0
        # return self.temp_done_counter > self.num_instances_per_env

    def _clean_actions(self, group, group_actions):
        '''
            Get the argmax when the Action Space is Discrete
            else,
                make sure it's numpy array
        '''
        # assert group_actions.shape == (self.BatchedStepResult[group].n_agents(), 1), "Actions for group {} should be of shape {}".format(group, (self.BatchedStepResult[group].n_agents(), 1))
        if self.GroupSpec[group].is_action_discrete():
            actions = np.array([ [np.argmax(_act)] for _act in group_actions ])
        elif type(group_actions) != np.ndarray:
            actions = np.array(group_actions)
        return actions

    def get_action_space_from_unity_spec(self, unity_spec):
        if unity_spec.is_action_discrete():
            return {
                'discrete': unity_spec.action_shape[0],
                'param': 0,
                'acs_space': unity_spec.action_shape[0]
            }
        elif unity_spec.is_action_continuous():
            return {
                'discrete': 0,
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

    def get_observation(self, group_ix):
        return list(self.observations.values())[group_ix]

    def get_actions(self):
        return list(self.actions.values())

    def get_action(self, group_ix):
        return self.actions[group_ix]

    def get_rewards(self):
        return list(self.rewards.values())

    def get_reward(self, group_ix):
        return list(self.rewards.values())[group_ix]

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

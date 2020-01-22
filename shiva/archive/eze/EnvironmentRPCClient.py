import grpc
import multiprocessing
import numpy as np

from shiva.envs.Environment import Environment
from shiva.envs.EnvironmentRPCServer import serve

from shiva.core.communication_objects.helpers_pb2 import Empty
from shiva.core.communication_objects.env_command_pb2 import EnvironmentCommand
from shiva.core.communication_objects.service_env_pb2_grpc import EnvironmentStub
from shiva.helpers.communications_helper import (
    from_action_to_EnvStepInput, from_EnvStepOutput_to_trajectories, from_EnvStepOutput_to_metrics
)

class EnvironmentRPCClient(EnvironmentStub):
    def __init__(self, channel, configs):
        super(EnvironmentRPCClient, self).__init__(channel)
        {setattr(self, k, v) for k, v in configs['Environment'].items()}
        self.configs = configs
        # self.__fake_super__(Environment, self.configs)

        env_specs = self.GetSpecs(Empty())
        self.action_space = {
            'discrete': env_specs.action_space.discrete,
            'param': env_specs.action_space.param,
            'acs_space': env_specs.action_space.acs_space
        }
        self.observation_space = env_specs.observation_space
        self.num_instances = env_specs.num_instances
        self.num_agents_per_instance = env_specs.num_agents_per_instance

        self.steps_per_episode = 0
        self.step_count = 0
        self.temp_done_counter = 0
        self.done_count = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0
        self.reward_total = 0

    def step(self, actions, *args, **kwargs):
        self.env_output = self.Step(from_action_to_EnvStepInput(actions))
        self.agent_ids = list(self.env_output.agent_states.keys())

        self.trajectory = from_EnvStepOutput_to_trajectories(self.env_output)
        self.metrics = from_EnvStepOutput_to_metrics(self.env_output)

        # Single Agent and Single Datapoint now on
        one_agent_id = self.agent_ids[0]
        self.observations, self.rewards_per_step, self.dones, self.extra = self.trajectory[one_agent_id][0]

        self.observations = np.array(self.observations)
        self.reward_per_episode = np.array(self.rewards_per_step)
        self.dones = np.array(self.dones, dtype=np.bool)

        self.steps_per_episode = self.metrics[one_agent_id].steps_per_episode
        self.step_count = self.metrics[one_agent_id].step_count
        self.temp_done_counter = self.metrics[one_agent_id].temp_done_counter
        self.done_count = self.metrics[one_agent_id].done_count
        self.reward_per_step = self.metrics[one_agent_id].reward_per_step
        self.reward_per_episode = self.metrics[one_agent_id].reward_per_episode
        self.reward_total = self.metrics[one_agent_id].reward_total

        return self.observations, self.reward_per_episode, self.done_count, self.extra

    def reset(self):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics.
            And maybe send a message???
        '''
        self.steps_per_episode = 0
        self.temp_done_counter = 0
        self.reward_per_step = 0
        self.reward_per_episode = 0

        self.env_output = self.Reset(Empty())
        self.trajectory = from_EnvStepOutput_to_trajectories(self.env_output)
        self.observations, self.rewards_per_step, self.dones, self.extra = self.trajectory['0'][0] # Single Agent, initial values!
        self.observations = np.array(self.observations)
        self.reward_per_episode = np.array(self.reward_per_episode)
        self.dones = np.array(self.dones, dtype=np.bool)

    def finished(self, n_episodes=None):
        '''
            The environment controls when the Learner is done stepping on the Environment
            Here we evaluate the amount of episodes to be played
        '''
        assert (n_episodes is not None), 'A @n_episodes is required to check if we are done running the Environment'
        return self.done_count >= n_episodes

    def is_done(self):
        '''
            One Shiva episode is when all instances in the Environment terminate at least once
        '''
        return self.dones
        # return self.temp_done_counter >= self.update_episodes

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

    def get_observation(self):
        return self.observations

    def get_current_step(self):
        return self.step_count

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
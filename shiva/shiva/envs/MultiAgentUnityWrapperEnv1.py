import torch
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from shiva.envs.Environment import Environment
from shiva.buffers.MultiTensorBuffer import MultiAgentTensorBuffer

class MultiAgentUnityWrapperEnv1(Environment):
    def __init__(self, config):
        # assert UnityEnvironment.API_VERSION == 'API-12', 'Shiva only support mlagents v12'
        self.on_policy = False
        super(MultiAgentUnityWrapperEnv1, self).__init__(config)
        self.start_unity_environment()
        self.set_initial_values()

    def start_unity_environment(self):
        self.channel = {
            'config': EngineConfigurationChannel(),
            'props': EnvironmentParametersChannel()
        }
        self.Unity = UnityEnvironment(
            file_name = self.exec,
            base_port = self.port if hasattr(self, 'port') else 5005, # 5005 is Unity's default value
            worker_id = self.worker_id,
            seed = self.worker_id * 5005,
            side_channels = [self.channel['config'], self.channel['props']],
            no_graphics = not self.render,
            # timeout_wait = self.timeout_wait if hasattr(self, 'timeout_wait') else 60
        )
        self.channel['config'].set_configuration_parameters(**self.unity_configs)
        # for k, v in self.unity_env_parameters.items():
        #     self.channel['props'].set_float_parameter(k, v)
        self.Unity.reset()
        self.log("Unity env started with behaviours: {}".format(self.Unity.get_behavior_names()))

    def set_initial_values(self):
        '''Unique Agent Behaviours'''
        self.roles = self.Unity.get_behavior_names()
        self.num_agents = len(self.roles)

        '''Grab all the Agents Specs'''
        self.RoleSpec = {role:self.Unity.get_behavior_spec(role) for role in self.roles}

        self.action_space = {role:self.get_action_space_from_unity_spec(self.RoleSpec[role]) for role in self.roles}
        self.observation_space = {role:self.get_observation_space_from_unity_spec(self.RoleSpec[role]) for role in self.roles}

        '''Init session cumulative metrics'''
        self.observations = {role:[] for role in self.roles}
        self.terminal_observations = {role:[] for role in self.roles}
        self.rewards = {role:[] for role in self.roles}
        self.dones = {role:[] for role in self.roles}
        self.reward_total = {role:0 for role in self.roles}

        # Collect first set of datas from environment
        self.DecisionSteps = {role:None for role in self.roles}
        self.TerminalSteps = {role:None for role in self.roles}
        self.trajectory_ready = {role:[] for role in self.roles}
        self.collect_step_data()
        '''Assuming all Roles Agent IDs are given at the beginning of the first episode'''
        self.role_agent_ids = {role:self.DecisionSteps[role].agent_id.tolist() for role in self.roles}

        '''Calculate how many instances Unity has'''
        self.num_instances_per_role = {role:len(self.DecisionSteps[role].agent_id) for role in self.roles}
        self.num_instances_per_env = self.DecisionSteps[self.roles[0]].obs[0].shape[0]

        '''Reset Metrics'''
        self.done_count = {role:0 for role in self.roles}
        self.steps_per_episode = {role:[0 for _ in self.role_agent_ids[role]] for role in self.roles}
        self.reward_per_step = {role:[0 for _ in self.role_agent_ids[role]] for role in self.roles}
        self.reward_per_episode = {role:[0 for _ in self.role_agent_ids[role]] for role in self.roles}

        self.metric_reset()

    def reset(self, force=False, *args, **kwargs):
        if force:
            self.Unity.reset()
            self.metric_reset(force=True)


    def metric_reset(self, force=False, *args, **kwargs):
        '''
            To be called by Shiva Learner
            It's just to reinitialize our metrics. Unity resets the environment on its own.
        '''
        self.temp_done_counter = 0
        for role in self.roles:
            for agent_id in self.role_agent_ids[role]:
                self.reset_agent_id(role, agent_id)
            if force:
                self.trajectory_ready[role] = []
                # maybe clear buffer?

    def reset_agent_id(self, role, agent_id):
        agent_ix = self.role_agent_ids[role].index(agent_id)
        self.steps_per_episode[role][agent_ix] = 0
        self.reward_per_step[role][agent_ix] = 0
        self.reward_per_episode[role][agent_ix] = 0

    def step(self, actions):
        self.raw_actions = {}
        self.actions = {}
        for ix, role in enumerate(self.roles):
            self.raw_actions[role] = actions[ix]
            self.actions[role] = self._clean_actions(role, actions[ix])
            self.Unity.set_actions(role, self.actions[role])
            self.log(f"{self.actions[role].shape} {self.actions[role]}", verbose_level=1)
        self.Unity.step()

        self.request_actions = False
        while not self.request_actions:
            self.collect_step_data()
            if not self.request_actions:
                # step until we get a DecisionStep (we need an action)
                self.Unity.step()
        '''
            Metrics collection
                Episodic # of steps             self.steps_per_episode
                Cumulative # of steps           self.step_count
                Temporary episode count         self.temp_done_counter
                Cumulative # of episodes        self.done_count
                Step Reward                     self.reward_per_step
                Episodic Reward                 self.reward_per_episode
                Cumulative Reward               self.reward_total (average upon all the agents that are controlled by that Role)
        '''
        self.temp_done_counter += sum(sum(self.dones[role]) for role in self.roles)

        '''Metrics are now updated withing self.collect_step_data()'''
        # for role in self.roles:
            # for agent_ix, role_agent_id in enumerate(self.role_agent_ids[role]):
                # self.steps_per_episode[role][agent_ix] += 1
                # self.reward_per_step[role][agent_ix] = self.rewards[role][agent_ix]
                # self.reward_per_episode[role][agent_ix] += self.rewards[role][agent_ix]
            # self.reward_total[role] += sum(self.rewards[role]) / self.num_instances_per_role[role] # total average reward
            # self.done_count[role] += sum(self.dones[role])

        return list(self.observations.values()), list(self.rewards.values()), list(self.dones.values()), {}

    def _unity_reshape(self, arr):
        '''Unity reshape of the data - concat all same Role agents trajectories'''
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, dim))

    def collect_step_data(self):
        """
        per @vincentpierre (MLAgents developer) - https://forum.unity.com/threads/decisionstep-vs-terminalstep.903227/#post-5927795
        If an AgentId is both in DecisionStep and TerminalStep, it means that the Agent reseted in Unity and immediately requested a decision.
        In your example, Agents 1, 7 and 9 had their episode terminated, started a new episode and requested a new decision. All in the same call to env.step()
        """

        for role in self.roles:
            self.DecisionSteps[role], self.TerminalSteps[role] = self.Unity.get_steps(role)
            # self.log(f"DecisionStep {self.DecisionSteps[role].agent_id}\nTerminalSteps {self.TerminalSteps[role].agent_id}")

            if len(self.TerminalSteps[role].agent_id) > 0:
                '''Agents that are on a Terminal Step'''
                self.terminal_observations[role] = self._flatten_observations(self.TerminalSteps[role].obs)
                self.trajectory_ready[role] = self.TerminalSteps[role].agent_id.tolist()
                for terminal_step_agent_ix, role_agent_id in enumerate(self.TerminalSteps[role].agent_id):
                    agent_ix = self.role_agent_ids[role].index(role_agent_id)
                    # Append to buffer last experience
                    exp = list(map(torch.clone, (torch.tensor([self.observations[role][agent_ix]], dtype=torch.float64),
                                                 torch.tensor([self.raw_actions[role][agent_ix]], dtype=torch.float64),
                                                 torch.tensor([self.TerminalSteps[role].reward[terminal_step_agent_ix]], dtype=torch.float64).unsqueeze(dim=-1),
                                                 torch.tensor([self.terminal_observations[role][terminal_step_agent_ix]], dtype=torch.float64),
                                                 torch.tensor([[True]], dtype=torch.bool).unsqueeze(dim=-1)
                                                 )))
                    self.trajectory_buffers[role][role_agent_id].push(exp)

                    # Update terminal metrics
                    self.done_count[role] += 1
                    self.steps_per_episode[role][agent_ix] += 1
                    self.reward_per_step[role][agent_ix] = self.rewards[role][agent_ix]
                    self.reward_per_episode[role][agent_ix] += self.rewards[role][agent_ix]
                    self.reward_total[role] = self.reward_total[role] + (self.reward_per_episode[role][agent_ix] / self.num_instances_per_role[role]) # Incremental Means Method

                    # Prepare trajectory for MPIEnv to send
                    self._ready_trajectories[role][role_agent_id] += [[*map(self._unity_reshape, self.trajectory_buffers[role][role_agent_id].all_numpy())] + [self.get_role_metrics(role, role_agent_id, episodic=True)]]
                    self.reset_agent_id(role, role_agent_id)
                    self.trajectory_buffers[role][role_agent_id].reset()

            if len(self.DecisionSteps[role].agent_id) > 0:
                '''Agents who need a next action'''
                # self.observations = {role:self._flatten_observations(self.DecisionSteps[role].obs) for role in self.roles} # unsure if needed to flatten observations???? This might be deprecated
                self.previous_observation = self.observations.copy()
                self.observations[role] = self._flatten_observations(self.DecisionSteps[role].obs)
                # self.log(f"{self.observations[role].shape}", verbose_level=1)
                self.rewards[role] = self.DecisionSteps[role].reward
                self.dones[role] = [False] * len(self.DecisionSteps[role].agent_id)
                if hasattr(self, 'raw_actions'):
                    self.request_actions = True

                    for decision_step_agent_ix, agent_id in enumerate(self.DecisionSteps[role].agent_id):
                        # if agent_id not in self.TerminalSteps[role].agent_id:
                        agent_ix = self.role_agent_ids[role].index(agent_id)
                        exp = list(map(torch.clone, (torch.tensor([self.previous_observation[role][agent_ix]], dtype=torch.float64),
                                                     torch.tensor([self.raw_actions[role][agent_ix]], dtype=torch.float64),
                                                     torch.tensor([self.DecisionSteps[role].reward[agent_ix]], dtype=torch.float64).unsqueeze(dim=-1),
                                                     torch.tensor([self.observations[role][agent_ix]], dtype=torch.float64),
                                                     torch.tensor([[False]], dtype=torch.bool).unsqueeze(dim=-1)
                                                     )))
                        self.trajectory_buffers[role][agent_id].push(exp)

                        # Update metrics at each agent step
                        self.steps_per_episode[role][agent_ix] += 1
                        self.reward_per_step[role][agent_ix] = self.rewards[role][agent_ix]
                        self.reward_per_episode[role][agent_ix] += self.rewards[role][agent_ix]

    def _flatten_observations(self, obs):
        '''Turns the funky (2, 16, 56) array into a (16, 112)'''
        return np.concatenate([o for o in obs], axis=-1)

    def get_metrics(self, episodic=True):
        '''MultiAgent Metrics'''
        metrics = {role:[self.get_role_metrics(role, role_agent_id, episodic) for role_agent_id in self.role_agent_ids[role]] for ix, role in enumerate(self.roles)}
        return list(metrics.values())

    def get_role_metrics(self, role, role_agent_id, episodic=True):
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step[role])
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode[role][role_agent_id]),
                (f'{role}/Steps_Per_Episode', self.steps_per_episode[role][role_agent_id])
            ]
        return metrics

    def is_done(self, n_episodes=None):
        '''
            Check if there's any role-agent that has finished the episode
        '''
        # self.log(f"DecisionSteps {self.DecisionSteps[self.roles[0]].agent_id}")
        # self.log(f"TerminalStep {self.TerminalSteps[self.roles[0]].agent_id}")
        return sum([len(self.trajectory_ready[role]) for role in self.roles]) > 0
        # return self.temp_done_counter >= self.num_instances_per_env

    def _clean_actions(self, role, role_actions):
        '''
            Get the argmax when the Action Space is Discrete
            else,
                make sure it's numpy array
        '''
        # assert group_actions.shape == (self.DecisionSteps[group].n_agents(), 1), "Actions for group {} should be of shape {}".format(group, (self.DecisionSteps[group].n_agents(), 1))
        if self.RoleSpec[role].is_action_discrete():
            actions = np.array([ [np.argmax(_act)] for _act in role_actions ])
        elif type(role_actions) != np.ndarray:
            actions = np.array(role_actions)
        # self.log(f"Clean action: {actions}")
        return actions

    def get_action_space_from_unity_spec(self, unity_spec):
        if unity_spec.is_action_discrete():
            if unity_spec.action_size > 1:
                # multi-discrete action branches to be implemented
                # For Unity example: In a game direction input(no movement, left, right) and jump input(no jump, jump)
                # there will be two branches(direction and jump), the first one with 3 options and the second
                # with 2 options (action_size = 2 and discrete_action_branches = (3, 2, ))
                raise NotImplemented
            else:
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

    def get_reward_episode(self, roles=False):
        episodic_reward = {}
        for role in self.roles:
            # take an average when there are many instances within one Unity simulation
            episodic_reward[role] = sum(self.reward_per_episode[role]) / len(self.reward_per_episode[role])
        return episodic_reward

    def get_rewards(self):
        return list(self.rewards.values())

    def create_buffers(self):
        '''
            Not best approach but solves current issue with Unity step function
            Need a buffer for each Agent as they terminate at different times and could turn the Tensor buffer into undesired shapes,
            so here we keep a separate buffer for each individual Agent in the simulation

            Secondary approach (possibly cheaper) could be to use a python list to collect data of current trajectory
            And convert to numpy before sending trajectory
            Test performance for Multiple environments within one single Unity simulation
        '''
        self.trajectory_buffers = {}
        self._ready_trajectories = {}
        for role in self.roles:
            self.trajectory_buffers[role] = {}
            self._ready_trajectories[role] = {}
            for role_agent_id in self.role_agent_ids[role]:
                self.trajectory_buffers[role][role_agent_id] = MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                                  1, #self.num_instances_per_role[role],
                                                                  self.observation_space[role],
                                                                  self.action_space[role]['acs_space'])
                self._ready_trajectories[role][role_agent_id] = []

    def reset_buffers(self):
        for _, role_buffers in self.trajectory_buffers.items():
            for _, role_agent_buffer in role_buffers.items():
                role_agent_buffer.reset()

    def close(self):
        self.Unity.close()
        delattr(self, 'Unity')

    def debug(self):
        try:
            print('self -->', self.__dict__)
            # print('UnityEnv -->', self.Unity.__dict__)
            print('RoleSpec -->', self.RoleSpec)
            print('BatchStepResults -->', self.batched_step_results.__dict__)
        except:
            print('debug except')
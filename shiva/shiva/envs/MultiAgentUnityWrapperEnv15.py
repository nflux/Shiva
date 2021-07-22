import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from shiva.buffers.MultiTensorBuffer import MultiAgentTensorBuffer
from shiva.envs.Environment import Environment


class MultiAgentUnityWrapperEnv15(Environment):
    """ Unity Wrapper that supports MLAgents release 15

    Args:
         config (dict): Expects a dictionary with the environment configurations.

    Note:
        Loads in the Unity binary, extracts information about the agents, establishes the connection,
        and sets up initial values in order to begin stepping in the environment.

    Returns:
        None
    """
    def __init__(self, config):
        assert UnityEnvironment.API_VERSION == '1.5.0', "Using wrong version of UnityEnvironment"
        super(MultiAgentUnityWrapperEnv15, self).__init__(config)
        self.start_unity_environment()
        self.set_initial_values()

    def start_unity_environment(self):
        """
        Loads in the binary and passes an configs and props to the Unity end.

        Returns:
            None
        """
        self.channel = {
            'config': EngineConfigurationChannel(),
            'props': EnvironmentParametersChannel()
        }
        np.random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        self.Unity = UnityEnvironment(
            file_name = self.exec,
            base_port = self.port if hasattr(self, 'port') else 5005, # 5005 is Unity's default value
            worker_id = self.worker_id,
            seed = self.manual_seed,
            side_channels = [self.channel['config'], self.channel['props']],
            no_graphics = not self.render,
            timeout_wait = self.timeout_wait if hasattr(self, 'timeout_wait') else 60
        )
        self.log(f"UnityEnv base_port {self.port} / worker_id {self.worker_id}", verbose_level=3)
        self.log(f"MANUAL SEED {self.manual_seed}")
        # https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md#environmentparameters
        self.channel['config'].set_configuration_parameters(**self.unity_configs)
        for param_name, param_value in self.unity_props.items():
            self.channel['props'].set_float_parameter(param_name, param_value)
        self.Unity.reset()

        if hasattr(self, 'skip_episodes') and self.skip_episodes > 0:
            self.log(f"Skipping {self.skip_episodes} episodes. Force restarting.")
            for i in range(self.skip_episodes):
                self.Unity.reset()

        self.log("Unity env started with behaviours: {}".format(list(self.Unity.behavior_specs)))

    def set_initial_values(self):
        """ Gets the environment ready to begin the simulation.

        Grabs the environment parameters, behaviors, and prepares dictionaries to store the DecisionSteps,
        TerminalSteps, and resets the metrics before beginning.

        Returns:
            None
        """
        '''Unique Agent Behaviours'''
        self.roles = list(self.Unity.behavior_specs)
        self.num_agents = len(self.roles)

        '''Grab all the Agents Specs'''
        self.RoleSpec = {role:self.Unity.behavior_specs[role] for role in self.roles}

        self.action_space = {role:self.get_action_space_from_unity_spec(self.RoleSpec[role]) for role in self.roles}
        self.observation_space = {role:self.get_observation_space_from_unity_spec(self.RoleSpec[role]) for role in self.roles}

        '''Init session cumulative metrics'''
        self.observations = {role:[] for role in self.roles}
        self.terminal_observations = {role:[] for role in self.roles}
        self.current_acs_mask = {role: [[]] for role in self.roles}
        self.prev_acs_mask = {role:[] for role in self.roles}
        self.rewards = {role:[] for role in self.roles}
        self.dones = {role:[] for role in self.roles}
        self.reward_total = {role:0 for role in self.roles}

        # Collect first set of datas from environment
        self.DecisionSteps = {role:None for role in self.roles}
        self.TerminalSteps = {role:None for role in self.roles}
        self.trajectory_ready_agent_ids = {role:[] for role in self.roles}

        self.collect_step_data() # collect first environment state
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
        """ Unity environment resets on it's own, but if `force` is True then will force restart and local metrics will restart as well.
        Args:
            force (bool): Default false. If True, Unity.reset() will be called and local metric collection will be restarted.
        Returns:
            None
        """
        if force:
            self.Unity.reset()
            self.metric_reset(force=True)

    def metric_reset(self, force=False, *args, **kwargs):
        """ Re-initializes our metrics. Unity resets the environment on its own; called by Shiva Learner.
        Args:
            force (bool): Hard reset.

        Returns:
            None
        """
        self.temp_done_counter = 0
        for role in self.roles:
            for agent_id in self.role_agent_ids[role]:
                self.reset_agent_id(role, agent_id)
            if force:
                self.trajectory_ready_agent_ids[role] = []
                # maybe clear buffer?

    def reset_agent_id(self, role, agent_id):
        """ Empties the data accumulators for a specifc BehaviorName

        Necessary whenever an agent's trajectory ends with either success or failure.

        Returns:
            None
        """
        agent_ix = self.role_agent_ids[role].index(agent_id)
        self.steps_per_episode[role][agent_ix] = 0
        self.reward_per_step[role][agent_ix] = 0
        self.reward_per_episode[role][agent_ix] = 0

    def step(self, actions):
        """ Steps in the Unity Environment

        Takes the actions given by the agent, processes them, and steps in the Unity Environment.

        Returns:
            A tuple of lists of Observations, Rewards, Done Flags, and an Empty Dict for Legacy reasons.
        """
        self.raw_actions = {}
        self.actions4Unity = {}
        # self.log(f"Raw Actions: {np.array(actions).shape}")
        for role_ix, role in enumerate(self.roles):
            self.raw_actions[role] = np.array(actions[role_ix]) # self._apply_action_masking(role, np.array(actions[role_ix]))
            # self.log(f"Role {role} Masked {self.raw_actions[role]}")
            self.actions4Unity[role] = self._clean_role_actions(role, self.raw_actions[role])
            self.Unity.set_actions(role, self.actions4Unity[role])
        # self.log(f"Before Unity.Step({self.steps_per_episode[self.roles[0]]})")
        self.Unity.step()
        # self.log(f"After Unity.Step({self.steps_per_episode[self.roles[0]]})")

        self.request_actions = False
        while not self.request_actions:
            self.collect_step_data()
            if not self.request_actions:
                # step until we get a DecisionStep (we need an action)
                self.Unity.step()

        # for i, r in enumerate(self.roles):
        #     if np.array_equal(self.previous_observation[r], self.observations[r]):
        #         self.log(f"Equality for {r}")

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

    def _apply_action_masking(self, role: str, role_actions: np.array) -> np.array:
        """
        Apply masking for each action branch of each agent of the given role.
        This places a 0 on the indexes for the actions that are not available.

        Args:
            role (str): role name
            role_actions (np.array): array whos dimension is (number of agents for this role, flat action dimension)

        Returns:
            np.array: actions with masking applied
        """
        if self.DecisionSteps[role].action_mask is None:
            return role_actions
        # Apply action masking and overwrite raw_action which will be stored in the buffer
        # this_role_action_mask = self.DecisionSteps[role].action_mask: List[np.array]
        # this_role_action_mask shape = (number of branches, number of agents for this role, action space masking)
        for agent_ix, agent_id in enumerate(self.role_agent_ids[role]):
            # this_agent_mask = this_role_action_mask[:, agent_ix, :]
            # self.log(f"Agent ID {agent_id} Mask {this_agent_mask}")
            _cum_ix = 0
            for branch_number, ac_dim in enumerate(self.RoleSpec[role].action_spec.discrete_branches):
                this_agent_branch_mask = self.DecisionSteps[role].action_mask[branch_number][agent_ix]
                # role_actions[agent_ix, _cum_ix:ac_dim+_cum_ix][this_agent_mask[branch_number]] = 0
                role_actions[agent_ix, _cum_ix:ac_dim+_cum_ix][this_agent_branch_mask] = 0
                _cum_ix += ac_dim
        return role_actions

    def _clean_role_actions(self, role: str, role_actions: np.array) -> np.array:
        """
        For discrete actions, it converts the action probabilities into discrete action selections (argmax) for each possible branch. For examples, [0.2, 0.7, 0.1] => [1]
        For continuous actions, it just converts it to numpy.

        Args:
            role (str): name of the role
            role_actions (np.array]): this array dimension should be of shape (number of agents for this role, action space)

        Returns:
            ActionTuple ready for Unity
        """
        UnityActionTuple = ActionTuple()
        if self.RoleSpec[role].action_spec.is_discrete():
            role_actions = np.array(role_actions)
            actions = np.zeros(shape=(len(self.role_agent_ids[role]), len(self.RoleSpec[role].action_spec.discrete_branches)))
            for agent_ix, agent_id in enumerate(self.role_agent_ids[role]):
                _cum_ix = 0
                for ac_ix, ac_dim in enumerate(self.RoleSpec[role].action_spec.discrete_branches):
                    actions[agent_ix, ac_ix] = self.action_selection(role_actions[agent_ix, _cum_ix:ac_dim + _cum_ix])
                    _cum_ix += ac_dim
            UnityActionTuple.add_discrete(actions)
            # self.log(f"Role {role} Cleaned {actions}", verbose_level=2)
        # elif type(role_actions) != np.ndarray:
        else:
            # actions = np.array(role_actions)
            UnityActionTuple.add_continuous(role_actions)
            # self.log(f"Role {role} Cleaned {role_actions}", verbose_level=2)
        # self.log(f"Clean action: {actions}")
        # return actions
        return UnityActionTuple

    def _unity_reshape(self, arr):
        """Unity reshape of the data - concats all same Role agents trajectories

        Reduces a three dimension array to two dimensions by squeezing the number of agents and the length
        of the trajectory into one dimension.

        Returns:
            Numpy array (total data values, dimension of data value)
        """
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, dim))

    def collect_step_data(self):
        """ Gets batches of data for the agents in the simulation.

        Note:
            Although this function doesn't return anything, the SARSA values are stored in internal data accumulator
            structures.

        Returns:
            None
        """
        """
        per @vincentpierre (MLAgents developer) - https://forum.unity.com/threads/decisionstep-vs-terminalstep.903227/#post-5927795
        If an AgentId is both in DecisionStep and TerminalStep, it means that the Agent reseted in Unity and immediately requested a decision.
        In your example, Agents 1, 7 and 9 had their episode terminated, started a new episode and requested a new decision. All in the same call to env.step()
        """
        for role in self.roles:
            # role_previous_acs_mask = self.get_current_action_masking(role)
            self.DecisionSteps[role], self.TerminalSteps[role] = self.Unity.get_steps(role)

            try:
                self.log(f"Role {role} Step {self.steps_per_episode[role]} Decision: {self.DecisionSteps[role].agent_id}, Terminal: {self.TerminalSteps[role].agent_id}", verbose_level=2)
                # self.log(f"CurrAcsMask {self.current_acs_mask[role]}")
            except:
                pass

            if len(self.TerminalSteps[role].agent_id) > 0:
                '''Agents that are on a Terminal Step'''
                self.log(f"TerminalSteps {self.TerminalSteps[role].agent_id}", verbose_level=3)
                self.terminal_observations[role] = self._flatten_observations(self.TerminalSteps[role].obs)
                self.trajectory_ready_agent_ids[role] += self.TerminalSteps[role].agent_id.tolist()
                for terminal_step_agent_ix, role_agent_id in enumerate(self.TerminalSteps[role].agent_id):
                    agent_ix = self.role_agent_ids[role].index(role_agent_id)

                    _last_step_reward = self.normalize_reward(self.TerminalSteps[role].reward[terminal_step_agent_ix]) if self.normalize else self.TerminalSteps[role].reward[terminal_step_agent_ix]

                    # Append to buffer last experience
                    exp = list(map(torch.clone, (torch.tensor([self.observations[role][agent_ix]], dtype=torch.float64),
                                                 torch.tensor([self.raw_actions[role][agent_ix]], dtype=torch.float64),
                                                 torch.tensor([_last_step_reward], dtype=torch.float64).unsqueeze(dim=-1),
                                                 torch.tensor([self.terminal_observations[role][terminal_step_agent_ix]], dtype=torch.float64),
                                                 torch.tensor([[True]], dtype=torch.bool).unsqueeze(dim=-1),
                                                 torch.tensor(self.prev_acs_mask[role][terminal_step_agent_ix, :]),
                                                 torch.tensor(self.get_empty_masks(role)[terminal_step_agent_ix, :])
                                                 )))
                    self.trajectory_buffers[role][role_agent_id].push(exp)

                    # Update terminal metrics
                    self.done_count[role] += 1
                    self.steps_per_episode[role][agent_ix] += 1
                    self.reward_per_step[role][agent_ix] = _last_step_reward # self.TerminalSteps[role].reward[terminal_step_agent_ix]
                    self.reward_per_episode[role][agent_ix] += _last_step_reward # self.TerminalSteps[role].reward[terminal_step_agent_ix]
                    self.reward_total[role] = self.reward_total[role] + (self.reward_per_episode[role][agent_ix] / self.num_instances_per_role[role]) # Incremental Means Method

                    # Prepare trajectory for MPIEnv to send
                    self._ready_trajectories[role][role_agent_id] += [[*map(self._unity_reshape, self.trajectory_buffers[role][role_agent_id].all_numpy())] + [self.get_role_metrics(role, role_agent_id, episodic=True)]]

                    # Reset
                    self.reset_agent_id(role, role_agent_id)
                    self.trajectory_buffers[role][role_agent_id].reset()

            # self.log(f"DecisionStep {self.DecisionSteps[role].reward} TerminalStep {self.TerminalSteps[role].reward}", verbose_level=-2)

            if len(self.DecisionSteps[role].agent_id) > 0:
                '''Agents who need a next action'''
                # self.observations = {role:self._flatten_observations(self.DecisionSteps[role].obs) for role in self.roles} # unsure if needed to flatten observations???? This might be deprecated
                self.previous_observation = self.observations.copy()
                self.observations[role] = self._flatten_observations(self.DecisionSteps[role].obs)
                self.current_acs_mask[role] = self.get_current_action_masking(role)
                # self.log(f"{self.observations[role].shape}", verbose_level=1)

                self.rewards[role] = self.normalize_reward(self.DecisionSteps[role].reward) if self.normalize else self.DecisionSteps[role].reward
                # self.rewards[role] = self.DecisionSteps[role].reward

                self.dones[role] = [False] * len(self.DecisionSteps[role].agent_id)

                '''If it's the first time we are collecting data, this IF statement will be False as we don't have an action to store into the buffer yet'''
                if hasattr(self, 'raw_actions'):
                    self.request_actions = True

                    for decision_step_agent_ix, role_agent_id in enumerate(self.DecisionSteps[role].agent_id):

                        # Important IF statement because if Agent_ID is on both DecisionStep and TerminalStep
                        # means that Unity automatically resetted the agent and it's on the very first state of the Env
                        # (where we haven't taken an action yet!)
                        if role_agent_id not in self.TerminalSteps[role].agent_id:
                            agent_ix = self.role_agent_ids[role].index(role_agent_id)
                            exp = list(map(torch.clone, (torch.tensor([self.previous_observation[role][agent_ix]], dtype=torch.float64),
                                                         torch.tensor([self.raw_actions[role][agent_ix]], dtype=torch.float64),
                                                         torch.tensor([self.rewards[role][agent_ix]], dtype=torch.float64).unsqueeze(dim=-1),
                                                         torch.tensor([self.observations[role][agent_ix]], dtype=torch.float64),
                                                         torch.tensor([[False]], dtype=torch.bool).unsqueeze(dim=-1),
                                                         torch.tensor(self.prev_acs_mask[role][decision_step_agent_ix, :]),
                                                         torch.tensor(self.current_acs_mask[role][decision_step_agent_ix, :])
                                                         )))
                            self.trajectory_buffers[role][role_agent_id].push(exp)

                            # Update metrics at each agent step
                            self.steps_per_episode[role][agent_ix] += 1
                            self.reward_per_step[role][agent_ix] = self.rewards[role][agent_ix]
                            self.reward_per_episode[role][agent_ix] += self.rewards[role][agent_ix]

            # Here we check if one of the Roles died and did not respawn: this means is not in DecisionStep nor TerminalStep.
            # Some environments dont respawn the agents immediately g.e. ICT Skirmish
            self.prev_acs_mask = self.current_acs_mask.copy()

    def _get_expected_action_shape(self, role: str) -> tuple:
        """Returns the expected shape of the action for the given role.

        Args:
            role (str): role name

        Returns:
            Tuple for the expected shape where index 0 is the number of agents for that role and index 1 is the expected action dimension for that agent.
        """
        return (len(self.Unity._env_state[role][0]), self.Unity._env_specs[role].action_size)

    def is_expecting_empty_action(self, role: str) -> bool:
        return self._get_expected_action_shape(role)[0] == 0

    def _flatten_observations(self, obs):
        """Turns the funky (2, 16, 56) array into a (16, 112)"""
        return np.concatenate([o for o in obs], axis=-1)

    def get_metrics(self, episodic=True):
        """MultiAgent Metrics

        Used for Tensorboard.

        Args:
            episodic (bool): Indicates whether to get episodic or stepwise metrics.

        Returns:
            A list of metric values
        """
        metrics = {role:[self.get_role_metrics(role, role_agent_id, episodic) for role_agent_id in self.role_agent_ids[role]] for ix, role in enumerate(self.roles)}
        return list(metrics.values())

    def get_role_metrics(self, role, role_agent_id, episodic=True):
        """ Gets the metrics for a specific role

        Used for Tensorboard.

        Args:
            role (str): BehaviorName of the agent.
            role_agent_id (int): Unique Agent Identifier.
            episodic (bool): Indicates whether to get episodic or stepwise metrics.
        Returns:
            A list of metric tuples.

        """
        agent_ix = self.role_agent_ids[role].index(role_agent_id)
        if not episodic:
            metrics = [
                ('Reward/Per_Step', self.reward_per_step[role][agent_ix])
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode[role][agent_ix]),
                (f'{role}/Steps_Per_Episode', self.steps_per_episode[role][agent_ix])
            ]
        return metrics

    def is_done(self, n_episodes=0):
        """Check if there's any role-agent that has finished the episode
        Args:
            n_episodes (int):
        """
        # self.log(f"DecisionSteps {self.DecisionSteps[self.roles[0]].agent_id}")
        # self.log(f"TerminalStep {self.TerminalSteps[self.roles[0]].agent_id}")
        return sum([len(self.trajectory_ready_agent_ids[role]) for role in self.roles]) > n_episodes

    def get_action_space_from_unity_spec(self, unity_spec):
        """ Checks BehaviorSpec (Agent) has a discrete or continuous actionspace.

        Used to infer the actionspace at runtime.

        Args:
            unity_spec (Dict): BehaviorName received from the unity environment.

        Returns:
            Discrete
                A dictionary of string, int pairs.
            Continuous
                A dictionary of string, tuple pairs.
        """
        print(unity_spec.action_spec, unity_spec.action_spec.is_discrete())
        if unity_spec.action_spec.is_discrete():
            action_space = {
                'discrete': unity_spec.action_spec.discrete_branches,
                'continuous': 0,
                'param': 0,
                'acs_space': unity_spec.action_spec.discrete_branches
            }
        elif unity_spec.action_spec.continuous_size > 0:
            action_space = {
                'discrete': 0,
                'continuous': (unity_spec.action_spec.continuous_size,),
                'param': 0,
                'acs_space': (unity_spec.action_spec.continuous_size,)
            }
        else:
            assert "Action space combining continuous and discrete space is not implemented"
        return action_space

    def get_observation_space_from_unity_spec(self, unity_spec):
        """ This sums up the obeservation_shapes for the agents in a given BehaviorSpec

        Returns:
            Integer
        """
        # flatten the obs_shape, g.e. from [(56,), (56,)] to 112
        return sum([ sum(obs_spec.shape) for obs_spec in unity_spec.observation_specs])

    def get_observations(self):
        """ Returns observations from the current step of each agent.

        Returns:
            list of np.arrays
        """
        return list(self.observations.values())

    def get_actions(self):
        """ Returns actions from the current step of each agent.

        Returns:
            list of np.arrays
        """
        return list(self.raw_actions.values())

    def get_current_action_masking(self, role: str) -> np.array:
        """Returned format is [number_of_agents_for_role, flattened action_space]"""
        m = self.DecisionSteps[role].action_mask
        if m == None:
            return self.get_empty_masks(role)
        else:
            return np.concatenate(self.DecisionSteps[role].action_mask, axis=-1)

    def get_empty_masks(self, role: str):
        num_instances_per_role = self.num_instances_per_role[role] if hasattr(self, 'num_instances_per_role') else len(self.DecisionSteps[role].agent_id)
        empty_mask = np.zeros((num_instances_per_role, sum(self.action_space[role]['acs_space']))).astype(np.bool)
        return empty_mask

    def get_reward_episode(self, roles=False):
        """ Returns the episodic rewards organized in a dictionary by role names.

        Returns:
            Dictionary with roles and floats as key value pairs
        """
        episodic_reward = {}
        for role in self.roles:
            # take an average when there are many instances within one Unity simulation
            episodic_reward[role] = sum(self.reward_per_episode[role]) / len(self.reward_per_episode[role])
        return episodic_reward

    def get_rewards(self):
        return list(self.rewards.values())

    def create_buffers(self):
        """ Makes buffers for episodic trajectories to be stored in.

        Todo:
            -Not best approach but solves current issue with Unity step function
             Need a buffer for each Agent as they terminate at different times and could turn the Tensor buffer into undesired shapes,
             so here we keep a separate buffer for each individual Agent in the simulation
            -Secondary approach (possibly cheaper) could be to use a python list to collect data of current trajectory
             And convert to numpy before sending trajectory
             Test performance for Multiple environments within one single Unity simulation

        Returns:
            None
        """
        self.trajectory_buffers = {}
        self._ready_trajectories = {}
        for role in self.roles:
            self.trajectory_buffers[role] = {}
            self._ready_trajectories[role] = {}
            for role_agent_id in self.role_agent_ids[role]:
                self.trajectory_buffers[role][role_agent_id] = MultiAgentTensorBuffer(self.episode_max_length+1, self.episode_max_length,
                                                                  1, #self.num_instances_per_role[role],
                                                                  self.observation_space[role],
                                                                  sum(self.action_space[role]['acs_space']))
                self._ready_trajectories[role][role_agent_id] = []

    def reset_buffers(self):
        """ Empties the trajectory buffers for the next episode.

        Returns:
            None
        """
        for _, role_buffers in self.trajectory_buffers.items():
            for _, role_agent_buffer in role_buffers.items():
                role_agent_buffer.reset()

    def close(self):
        """ Closes the connection with the Unity Environment.
        Also deletes the environment attribute from the Class.
        Returns:
            None
        """
        self.Unity.close()
        delattr(self, 'Unity')

    def debug(self):
        """ Prints out Behavior Names, and Batched Steps

        Returns:
            None
        """
        try:
            print('self -->', self.__dict__)
            # print('UnityEnv -->', self.Unity.__dict__)
            print('RoleSpec -->', self.RoleSpec)
            print('BatchStepResults -->', self.batched_step_results.__dict__)
        except:
            print('debug except')

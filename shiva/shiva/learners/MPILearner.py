import sys, traceback, os, time, pickle
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch, time
torch.set_printoptions(profile="full")
import numpy as np
from scipy import stats
from collections import deque
from mpi4py import MPI

from shiva.helpers.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.core.IOHandler import get_io_stub
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process, flat_1d_list
from shiva.learners.Learner import Learner


class MPILearner(Learner):

    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    def __init__(self):
        """
        Learner implementation for a distributed architecture where we can enable Population Based Training.
        """
        # Receive Config from Meta
        self.configs = self.meta.scatter(None, root=0)
        self.set_default_configs()
        super(MPILearner, self).__init__(self.id, self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())), verbose_level=1)
        self.log("Received config {}".format(self.configs), verbose_level=3)
        self.launch()

    def launch(self):
        """
        Initialization function.
        * Connects with the IOHandler
        * Creates the MPI port where trajectories are gonna arrived (from the MPIEnv)
        * Initializes algorithm, creates agents and replay buffer
        * Hand shake with the Meta Learner
        * Connects with MultiEnv (soon to be deprecated as there's no communication going on)

        Returns:
            None
        """
        self._connect_io_handler()

        '''Process Configs'''
        # Assuming all Environments running are of the same type
        self.menvs_specs = self.configs['MultiEnv']
        self.env_specs = self.menvs_specs[0]['env_specs']
        self.num_envs = self.configs['Environment']['num_envs']
        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']

        # Create Port for each single Environment process group
        self.port = {'env': []}
        for i in range(self.num_menvs):
            self.port['env'].append(MPI.Open_port(MPI.INFO_NULL))

        '''Assuming Learner has 1 Agent per Role'''
        self.num_agents = len(self.roles)
        self.observation_space = self.env_specs['observation_space']
        self.action_space = self.env_specs['action_space']
        self.log("Obs space {} / Action space {}".format(self.observation_space, self.action_space), verbose_level=2)

        # Initialize inter components
        self.alg = self.create_algorithm()
        self.buffer = self.create_buffer()
        self.agents = self.create_agents()

        # Check in with Meta
        self.meta.gather(self._get_learner_specs(), root=0)

        # make first saving
        Admin.checkpoint(self, checkpoint_num=0, function_only=True, use_temp_folder=True)

        # Connect with MultiEnvs
        self._connect_menvs()
        self.log("Running..", verbose_level=1)
        self.run()

    def run(self):
        """
        Training loop where we execute high level functions:
        * Check and receive trajectories
        * Run algorithmic updates
        * Run evolution (if PBT enabled)
        * Plot metrics on Tensorboard (using ShivaAdmin)

        Returns:
            None
        """
        self.step_count = {agent.id:0 for agent in self.agents}
        self.done_count = 0
        self.num_updates = 0
        self.steps_per_episode = {agent.id:0 for agent in self.agents}
        self.metrics_env = {agent.id:[] for agent in self.agents}
        self.last_rewards = {agent.id:{'q':deque(maxlen=self.configs['Agent']['lr_decay']['average_episodes']), 'n':0} for agent in self.agents}

        self.profiler.start(["AlgUpdates", 'ExperienceReceived'])
        # while self.done_count < self.episodes:
        while True:
            self.check_incoming_trajectories()
            self.run_updates()
            self.run_evolution()
            self.collect_metrics(episodic=True) # tensorboard

        self.close()

    def check_incoming_trajectories(self):
        """
        This function iterates for `n_traj_pulls`_ times over all the MPIEnv groups and tries to pull a trajectory from their queue if there is any there. Also profiles time spent between pulls and logs.

        Returns:
            None
        """
        self.last_metric_received = None

        self._n_success_pulls = 0
        self.metrics_env = {agent.id:[] for agent in self.agents}
        for _ in range(self.n_traj_pulls):
            for comm in self.envs:
                self.receive_trajectory_numpy(comm)
        if self._n_success_pulls > 0:
            self.profiler.time('ExperienceReceived', self.done_count, output_quantity=self._n_success_pulls)
        if self.last_metric_received is not None: # and self.done_count % 20 == 0:
            self.log("{} {}:{}".format(self._n_success_pulls, self.done_count, self.last_metric_received), verbose_level=1)

    def receive_trajectory_numpy(self, env_comm):
        """
        If the MPI queue has an available trajectory to be pulled we are gonna pull it and push it into the local replay buffer.

        Args:
            env_comm (MPI.Comm): MPI communication object. For more information go to mpi4py docs

        Returns:
            None
        """        # env_comm = self.envs
        if env_comm.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=self.info):
            env_source = self.info.Get_source()
            self.traj_info = env_comm.recv(None, source=env_source, tag=Tags.trajectory_info) # block statement

            traj_length_index = self.traj_info['length_index']
            traj_length = self.traj_info['obs_shape'][traj_length_index]
            # assert role == self.roles, "<Learner{}> Got trajectory for {} while we expect for {}".format(self.id, role, self.roles)
            assert set(self.roles).issuperset(set(self.traj_info['role'])), "<Learner{}> Got trajectory for {} while we expect for {}".format(self.id, self.traj_info['role'], self.roles)

            observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
            env_comm.Recv([observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_observations)

            actions = np.empty(self.traj_info['acs_shape'], dtype=np.float64)
            env_comm.Recv([actions, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_actions)

            rewards = np.empty(self.traj_info['rew_shape'], dtype=np.float64)
            env_comm.Recv([rewards, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_rewards)

            next_observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
            env_comm.Recv([next_observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_next_observations)

            dones = np.empty(self.traj_info['done_shape'], dtype=np.float64)
            env_comm.Recv([dones, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_dones)

            actions_mask = np.empty(self.traj_info['acs_shape'], dtype=np.bool)
            env_comm.Recv([actions_mask, MPI.BOOL], source=env_source, tag=Tags.trajectory_actions_mask)

            next_actions_mask = np.empty(self.traj_info['acs_shape'], dtype=np.bool)
            env_comm.Recv([next_actions_mask, MPI.BOOL], source=env_source, tag=Tags.trajectory_next_actions_mask)

            # self.step_count += traj_length
            self.done_count += 1
            # self.steps_per_episode = traj_length
            # below metric could be a potential issue when Learner controls multiple agents and receives individual trajectories for them

            # self.reward_per_episode = sum(rewards.squeeze())
            # rs = rewards.squeeze()
            # if (rs.size != 1):
            #     self.reward_per_episode = sum(rs)
            # else:
            #     self.reward_per_episode = rs
            # print(rs.shape)
            # print(rewards.shape)

            self._n_success_pulls += 1
            self.log("Got TrajectoryInfo\n{}".format(self.traj_info), verbose_level=3)

            for role_ix, role in enumerate(self.traj_info['role']):
                reward_per_episode = sum(rewards[role_ix, :, 0]).item()
                '''Assuming 1 agent per role here'''
                agent_id = self.role2ids[role][0]
                self.step_count[agent_id] += traj_length
                self.steps_per_episode[agent_id] = traj_length
                for metric_ix, metric_set in enumerate(self.traj_info['metrics'][role_ix]):
                    self.traj_info['metrics'][role_ix][metric_ix] += (self.done_count,) # add the x_value for tensorboard!
                self.metrics_env[agent_id] += self.traj_info['metrics'][role_ix]
                self.last_rewards[agent_id]['q'].append(reward_per_episode)

            self.last_metric_received = f"{self.traj_info['env_id']} got ObsShape {observations.shape} {self.traj_info['metrics']}"

            self.log(f"Obs {observations.shape} {observations}", verbose_level=3)
            self.log(f"Acs {actions}", verbose_level=3)
            self.log(f"AcsMask {actions_mask}", verbose_level=3)
            self.log(f"NextAcsMask {next_actions_mask}", verbose_level=3)
            # self.log(f"Rew {rewards.shape} {rewards}")
            # self.log(f"NextObs {next_observations}")
            # self.log(f"Dones {dones}")

            '''VERY IMPORTANT Assuming each individual role has same acs/obs dimension and reward function'''
            exp = list(map(torch.clone, (torch.from_numpy(observations).reshape(traj_length, len(self.traj_info['role']), observations.shape[-1]),
                                         torch.from_numpy(actions).reshape(traj_length, len(self.traj_info['role']), actions.shape[-1]),
                                         torch.from_numpy(rewards).reshape(traj_length, len(self.traj_info['role']), rewards.shape[-1]),
                                         torch.from_numpy(next_observations).reshape(traj_length, len(self.traj_info['role']), next_observations.shape[-1]),
                                         torch.from_numpy(dones).reshape(traj_length, len(self.traj_info['role']), dones.shape[-1]),
                                         torch.from_numpy(actions_mask).reshape(traj_length, len(self.traj_info['role']), actions.shape[-1]),
                                         torch.from_numpy(next_actions_mask).reshape(traj_length, len(self.traj_info['role']), actions.shape[-1]),
                                         )))

            # rewarding_ixs = []
            # value = 5.0
            # for i in range(rewards.shape[1]):
            #     if rewards[0, i, 0] == value:
            #         rewarding_ixs += [i]
            #
            #
            # obs_ixs_with_1 = []
            # next_obs_ixs_with_1 = []
            # for ix in rewarding_ixs:
            #     for j in range(next_observations.shape[2]):
            #         if observations[0, ix, j] == 1.0:
            #             obs_ixs_with_1 += [j]
            #         if next_observations[0, ix, j] == 1.0:
            #             next_obs_ixs_with_1 += [j]
            #
            # self.log(rewards)
            # self.log(f"Found Rewarding IXs {rewarding_ixs}")
            # self.log(observations[0, rewarding_ixs, :])
            # self.log(rewards[0, rewarding_ixs, :])
            # self.log(next_observations[0, rewarding_ixs, :])
            #
            # self.log(f"Observation IXs where there is a 1: {obs_ixs_with_1}")
            # self.log(f"NextObs IXs where there is a 1: {next_obs_ixs_with_1}")

            self.buffer.push(exp)

    def run_updates(self):
        """
        The core training occurs here where we call `algorithm.update()`.
        Before updating performs Meta Learning by decaying learning rate if necessary. Also saves new checkpoints.

        Returns:
            None
        """
        if not self.evaluate \
                and (self.done_count % self.episodes_to_update == 0) \
                and len(self.buffer) > self.buffer.batch_size:

            if not self.evolve:
                # Might need to have another criteria for when to decay LR when using PBT
                self.run_recalculate_hyperparameters()
            elif self.done_count % self.configs['Agent']['lr_decay']['average_episodes'] == 0:
                self.log(f"ID {self.agents[0].id} Last{self.configs['Agent']['lr_decay']['average_episodes']}AveRew {sum(self.last_rewards[self.agents[0].id]['q']) / len(self.last_rewards[self.agents[0].id]['q'])}")

            self.alg.update(self.agents, self.buffer, self.done_count, episodic=True)
            self.num_updates = self.alg.get_num_updates()
            self.profiler.time('AlgUpdates', self.num_updates, output_quantity=self.alg.update_iterations)

            '''Save latest updated agent in temp folder for MultiEnv and Evals to load'''
            self.checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)

            '''Check point purposes only'''
            if self.done_count >= self.checkpoints_made * self.save_checkpoint_episodes:
                self.checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=False)

            for agent in self.agents:
                agent.step_count = self.step_count[agent.id]
                agent.done_count = self.done_count
                agent.num_updates = self.num_updates

    def run_recalculate_hyperparameters(self):
        """
        Check if hyperparameters need to be updated (decay or restore).
        See Learner config explanation for more details on usage.

        Returns:
            None
        """
        _decay_or_restore_lr = 0  # -1 to decay, 1 to restore, 0 to do nothing
        _decay_log = ""
        for agent in self.agents:
            # update this HPs so that they show up on tensorboard
            agent.recalculate_hyperparameters()
            if 'lr_decay' in self.configs['Agent']:
                if not agent.is_exploring() and self.done_count > (self.configs['Agent']['lr_decay']['wait_episodes_to_decay'] * self.last_rewards[agent.id]['n']):
                    self.last_rewards[agent.id]['n'] = self.done_count // self.configs['Agent']['lr_decay']['wait_episodes_to_decay'] + 1
                    agent_ave_reward = sum(self.last_rewards[agent.id]['q']) / len(self.last_rewards[agent.id]['q'])
                    if self.configs['Environment']['expert_reward_range'][agent.role][0] <= agent_ave_reward <= self.configs['Environment']['expert_reward_range'][agent.role][1]:
                        agent.decay_learning_rate()
                        _decay_or_restore_lr = -1
                        try:
                            _decay_log += f"Decay Actor {agent.id} LR {agent.actor_learning_rate} with Last{self.configs['Agent']['lr_decay']['average_episodes']}AveRew {agent_ave_reward}\n"
                        except:
                            pass
                    else:
                        agent.restore_learning_rate()
                        _decay_or_restore_lr = 1
                        try:
                            _decay_log += f"Restore Actor {agent.id} LR {agent.actor_learning_rate} with Last{self.configs['Agent']['lr_decay']['average_episodes']}AveRew {agent_ave_reward}\n"
                        except:
                            pass
        try:
            self.alg.decay_learning_rate() if _decay_or_restore_lr == -1 else self.alg.restore_learning_rate() if _decay_or_restore_lr == 1 else None
        except:
            pass
        # self.log(f"{_decay_log}\nCentralCritic LR {self.alg.critic_learning_rate}", verbose_level=1)# if _decay_log != '' else None

    def run_evolution(self):
        """
        This functions is only executed when PBT is enabled. It will check if it's time to evolve and will ask a evolution config to the Meta Learner.
        Once a new evolution config is received it will proceed with the evolution procedures for exploitation and exploration given by the Meta Learner.

        Returns:
            None
        """
        '''
            Expectation value of how many parameter to change per evolution = 1
        '''
        if self.pbt and self.done_count >= self.initial_evolution_episodes:

            if (self.done_count - self.initial_evolution_episodes) >= (self.n_evolution_requests * self.evolution_episodes):
                self.meta.send(self._get_learner_specs(), dest=0, tag=Tags.evolution_request) # ask for evolution configs1
                self.n_evolution_requests += 1
                # self.log("Ask for Evolution", verbose_level=3)

            if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution_config, status=self.info):
                self.evolution_config = self.meta.recv(None, source=self.info.Get_source(), tag=Tags.evolution_config)  # block statement

                if not self.evolve:
                    self.log("self.evolve is set to {}! Got Evolution {}".format(self.evolve, self.evolution_config), verbose_level=1)
                    return

                self.log('Got Evolution {}'.format(self.evolution_config), verbose_level=1)
                self.evolution_count += 1

                for evol_config in self.evolution_config:
                    if evol_config['evolution'] == False or evol_config['agent_id'] == evol_config['evo_agent_id']:
                        continue
                    agent = self.get_agent_of_id(evol_config['agent_id'])
                    setattr(self, 'exploitation', getattr(self, evol_config['exploitation']))
                    setattr(self, 'exploration', getattr(self, evol_config['exploration']))
                    self.exploitation(agent, evol_config)
                    self.exploration(agent)

                '''No need to send message to MultiEnv for now'''
                # for ix in range(self.num_menvs):
                #     self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

                # self.log('Evolution Completed for {} agents'.format(len(self.agents)), verbose_level=2)
            # else:
            #     self.log("No configs probed!", verbose_level=1)

    def create_agents(self):
        """
        Initialization function where we create the agents or load the agents for continue training or to be evaluated.

        Returns:
            List[Agent]
        """
        assert hasattr(self, 'num_agents') and self.num_agents > 0, 'Learner num_agent not specified, got {}'.format(self.num_agents)
        self.start_agent_idx = (self.id+1) * 1000
        self.new_agents_ids = np.arange(self.start_agent_idx, self.start_agent_idx + self.num_agents)

        if self.load_agents:
            agents = Admin.load_agents(self.load_agents, absolute_path=False, load_latest=False, device=self.alg.device) # the alg determines the device
            # minor tweak on the agent id as we can have an issue when multiple learners load the same agent id (like loading a PBT session)
            for a in agents:
                a.id += 10 * self.id
            self.alg.add_agents(agents)
            agent_creation_log = "{agents_strs} agents loaded"
        elif hasattr(self, 'roles') and len(self.roles) > 0:
            self.agents_dict = {role:self.alg.create_agent_of_role(self.new_agents_ids[ix], role) for ix, role in enumerate(self.roles)}
            agents = list(self.agents_dict.values())
            agent_creation_log = "{agents_strs} agents created"
        else:
            agents = [self.alg.create_agent(ix) for ix in self.new_agents_ids]
            agent_creation_log = "{agents_strs} agents created"

        self.role2ids = {role:[] for role in self.roles}
        self.id2role = {}
        self.metrics_env = {}
        for _agent in agents:
            if 'roles_remap' in self.configs['Environment']:
                _agent.role = self.configs['Environment']['roles_remap'][_agent.role]
            _agent.to_device(self.alg.device)
            _agent.evaluate = self.evaluate
            self.role2ids[_agent.role] += [_agent.id]
            self.metrics_env[_agent.id] = []
            self.id2role[_agent.id] = _agent.role
            _agent.recalculate_hyperparameters()

        self.log(agent_creation_log.format(agents_strs=[str(a) for a in agents]), verbose_level=1)
        return agents

    def get_agent_of_id(self, id):
        """
        For a given Agent ID, returns the Agent.

        Args:
            id (int): Agent ID

        Returns:
            Agent
        """
        for agent in self.agents:
            if agent.id == id:
                return agent

    def create_algorithm(self):
        """
        Initialization function.

        Returns:
            Algorithm
        """
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        self.configs['Algorithm']['roles'] = self.roles if hasattr(self, 'roles') else []
        if 'device' in self.configs['Algorithm']:
            if type(self.configs['Algorithm']['device']) == list:
                # A list of GPUs was given for the Learners/Algorithm, choose one
                gpu_ix = int(self.id % len(self.configs['Algorithm']['device']))
                gpu_chosen = self.configs['Algorithm']['device'][gpu_ix]
                self.configs['Algorithm']['device'] = gpu_chosen
        else:
            self.configs['Algorithm']['device'] = torch.device("gpu" if torch.cuda.is_available() else "cpu")
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Created ".format(str(alg)), verbose_level=-1)
        return alg

    def create_buffer(self):
        """
        Initialization function

        Returns:
            ReplayBuffer
        """
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if type(self.action_space) == dict:
            '''Assuming roles with same obs/acs dim'''
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'],
                                  self.num_agents, self.observation_space[self.roles[0]],
                                  sum(self.action_space[self.roles[0]]['acs_space']))
        else:
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.log("Buffer created of type {}".format(buffer_class), verbose_level=2)
        return buffer

    def _connect_io_handler(self):
        """
        Connect to the gRPC IOHandler. See `get_io_stub` helper function to get more depth information on it's work.

        Returns:
            None
        """
        self.io = get_io_stub(self.configs)

    def checkpoint(self, checkpoint_num, function_only, use_temp_folder):
        """
        Performs the checkpoint saving using ShivaAdmin.

        Args:
            checkpoint_num (int): current checkpoint number
            function_only (bool): if we want to use ShivaAdmin as a function_only helper. Used for distributed architecture!
            use_temp_folder (bool): If True, checkpoints will be saved in the 'last' folder. If False, a new checkpoint folder will be created.

        Returns:
            None
        """
        self.checkpoints_made += 1 if not use_temp_folder else 0

        for a in self.agents:
            self.alg.save_central_critic(a)

        if use_temp_folder:
            self.io.request_io(self._get_learner_specs(), Admin.get_learner_url(self), wait_for_access=True)
        Admin.checkpoint(self, checkpoint_num=checkpoint_num, function_only=function_only, use_temp_folder=use_temp_folder)
        if use_temp_folder:
            self.io.done_io(self._get_learner_specs(), Admin.get_learner_url(self))

    def t_test(self, agent, evo_config):
        """

        Args:
            agent (Agent): Agent that is being tested with the `t_test` metric. If `t_test` passes, Agent will be truncated with the Evo Agent.
            evo_config (Dict): evolution config. For more information look at the Meta Learner generating evolution configs.

        Returns:
            None
        """
        if evo_config['ranking'] > evo_config['evo_ranking']:

            my_eval_path = f"{Admin.get_learner_url(self)}/evaluations/"
            self.io.request_io(self._get_learner_specs(), my_eval_path, wait_for_access=True)
            evals = np.load(f"{my_eval_path}Agent_{evo_config['agent_id']}.npy")
            self.io.done_io(self._get_learner_specs(), my_eval_path)

            evo_path = f"{evo_config['evo_path']}/evaluations/"
            self.io.request_io(self._get_learner_specs(), evo_path, wait_for_access=True)
            evo_evals = np.load(f"{evo_path}Agent_{evo_config['evo_agent_id']}.npy")
            self.io.done_io(self._get_learner_specs(), evo_path)

            if self.welch_T_Test(evals, evo_evals):
                self.truncation(agent, evo_config)

    def welch_T_Test(self, evals, evo_evals):
        """
        Performs the `t_test` between the rewards distribution of our local agent and the evo agent.

        Args:
            evals (np.ndarray): rewards obtained by our Agent on the last evaluation run
            evo_evals (np.ndarray): rewards obtained by the Evo Agent on the last evaluation run

        Returns:
            bool: indicating if the `p` value between both reward distributions is less than the config `p_value`
        """
        t, p = stats.ttest_ind(evals, evo_evals, equal_var=False)
        self.log(f"T_test {evals} and {evo_evals}", verbose_level=3)
        self.log(f"T_Test P_value {p}", verbose_level=3)
        return p < self.p_value

    def truncation(self, agent, evo_config):
        """
        Performs truncation. Loads the Evo Agent where our Agent is gonna truncate from.

        Args:
            agent (Agent): Agent being truncated
            evo_config (Dict): Evolution config containing path where we can load the Evo Agent

        Returns:
            None
        """
        self.io.request_io(self._get_learner_specs(), evo_config['evo_path'], wait_for_access=True)
        evo_agent = Admin.load_agent_of_id(evo_config['evo_path'], evo_config['evo_agent_id'])[0]
        self.io.done_io(self._get_learner_specs(), evo_config['evo_path'])

        agent.copy_hyperparameters(evo_agent)
        self.alg.copy_hyperparameters(evo_agent)

        agent.copy_weights(evo_agent)
        self.alg.copy_weight_from_agent(evo_agent)

    def perturb(self, agent):
        """
        Performs hyperparameter perturbation. Note that when using a central critic algorithm we also call the Algorithm.

        Args:
            agent (Agent): Agent who we are perturbing the hyperparameters.

        Returns:
            None
        """
        perturb_factor = np.random.choice(self.perturb_factor)
        agent.perturb_hyperparameters(perturb_factor)
        self.alg.perturb_hyperparameters(perturb_factor)

    def resample(self, agent):
        """
        Performs hyperparameters resampling. Note that when using a central critic algorithm we also call the Algorithm.

        Args:
            agent (Agent): Agent who we are resampling hyperparameters for.

        Returns:
            None
        """
        agent.resample_hyperparameters()
        self.alg.resample_hyperparameters()

    def exploitation(self):
        raise NotImplemented

    def exploration(self):
        raise NotImplemented

    def _connect_menvs(self):
        """
        Connects with the MPIMultiEnv.

        Returns:
            None
        """
        # Connect with MultiEnv
        self.menv = MPI.COMM_WORLD.Connect(self.menv_port,  MPI.INFO_NULL)
        self.log('Connected with MultiEnvs', verbose_level=2)

        for i in range(self.num_menvs):
            self.menv.send(self._get_learner_specs(), dest=i, tag=Tags.specs)

        # We need to accept the connection from each Environments group == number of MultiEnvs
        self.envs = []
        for env_port in self.port['env']:
            self.envs.append(MPI.COMM_WORLD.Accept(env_port))
        # self.envs = MPI.COMM_WORLD.Accept(self.port)
        self.log("Connected with {} Env Groups".format(len(self.envs)), verbose_level=2)

    def _get_learner_state(self):
        specs = self._get_learner_specs()
        specs['num_updates'] = self.num_update
        return specs

    def _get_learner_specs(self):
        """
        Creates a new Learner spec depending on the current state of the Learner.

        Returns:
            Dict
        """
        return {
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles,
            'num_agents': self.num_agents,
            'agent_ids': list([a.id for a in self.agents]) if hasattr(self, 'agents') else None,
            'role2ids': self.role2ids,
            'done_count': self.done_count if hasattr(self, 'done_count') else None,
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_learner_url(self),
        }

    def get_metrics(self, episodic, agent_id):
        """
        Collects the metrics from
        * Agent
        * Environment (received with the trajectory)

        Args:
            agent_id (int): Agent ID for which we want to get the metrics from.

        Returns:
            List[Union[List[Tuple[str, float, int]], Tuple[str, float, int]]]: list of metrics or list of list of metrics
        """
        evolution_metrics = []
        # if self.pbt:
        agent = self.get_agent_of_id(agent_id)
        evolution_metrics += agent.get_metrics()
        # evolution_metrics += [('Agent/{}/Actor_Learning_Rate'.format(agent.role), agent.actor_learning_rate)]
        return [self.metrics_env[agent_id]] + evolution_metrics

    def set_default_configs(self):
        """
        Cleans a bit the config and set some default values.

        Returns:
            None
        """
        assert 'Learner' in self.configs, 'No Learner config found on received config: {}'.format(self.configs)
        default_configs = {
            'evaluate': False,
            'n_traj_pulls': 1,
            'episodes': float('inf')
            # 'perturb_factor': [0.8, 1.2]
            # add default configs here

        }
        for attr_name, default_val in default_configs.items():
            if attr_name not in self.configs['Learner']:
                self.configs['Learner'][attr_name] = default_val

        assert 'Agent' in self.configs, "No Agent config given!, got {}".format(self.configs)
        default_agent_configs = {
            'lr_decay': {'factor': 1, 'average_episodes': 100, 'wait_episodes_to_decay': 100} # no decay by default
        }
        for attr_name, default_val in default_agent_configs.items():
            if attr_name not in self.configs['Agent']:
                self.configs['Agent'][attr_name] = default_val

        assert 'Algorithm' in self.configs, "No Algorithm config given~, got {}".format(self.configs)
        default_algorithm_configs = {
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        }
        for attr_name, default_val in default_algorithm_configs.items():
            if attr_name not in self.configs['Algorithm']:
                self.configs['Algorithm'][attr_name] = default_val

    def close(self):
        """
        Notify the Meta Learner that we are closing, and then close all connections.

        Returns:
            None
        """
        self.log("Started closing", verbose_level=2)
        self.meta.send(self._get_learner_specs(), dest=0, tag=Tags.close)
        for e in self.envs:
            e.Disconnect()
        # Close Environment port
        for portname in self.port['env']:
            MPI.Close_port(portname)
        self.log("Closed Environments", verbose_level=2)
        self.menv.Disconnect()
        self.log("Closed MultiEnv", verbose_level=2)
        self.meta.Disconnect()
        self.log("FULLY CLOSED", verbose_level=1)
        exit(0)

    def __str__(self):
        return "<Learner(id={})>".format(self.id)


if __name__ == "__main__":
    try:
        l = MPILearner()
    except Exception as e:
        msg = "<Learner(id={})> error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
        terminate_process()
    finally:
        pass
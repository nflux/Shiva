import sys, time, traceback, subprocess, torch
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.core.IOHandler import get_io_stub
from shiva.envs.Environment import Environment
from shiva.helpers.misc import terminate_process, flat_1d_list

class MPIMultiEnv(Environment):

    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    def __init__(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEnv, self).__init__(self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())), verbose_level=1)
        Admin.init(self.configs)
        self.launch()

    def launch(self):
        self._connect_io_handler()

        if hasattr(self, 'device') and self.device == 'gpu':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')

        # Open Port (for Learners)
        self.port = MPI.Open_port(MPI.INFO_NULL)

        self._launch_envs()

        # Handshake with Meta to send Environment Specs and my Port for Learners to connect
        self.meta.gather(self._get_menv_specs(), root=0)

        self._connect_learners()
        self._receive_match(bypass_request=True) # receive first match, no IO request for the first one to avoid overhead

        # Prepare
        self.step_count = 0
        self.done_count = 0
        '''
            Check if we can do a numpy step instead of python list
            If obs/acs dimensions for all roles are the same, then we can do MPI
        '''
        if 'Unity' in self.type or 'ParticleEnv' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0] ), dtype=np.float64)
        elif 'Gym' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space'] ), dtype=np.float64)
        elif 'RoboCup' in self.type:
            self._obs_recv_buffer = np.empty((self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space']), dtype=np.float64)

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)

        self.run()

    def run(self):
        self._time_to_load = False
        self.is_running = True
        while self.is_running:
            time.sleep(self.configs['Admin']['time_sleep']['MultiEnv'])
            self._step_python()
            # self._step_numpy()
            self.check_state()
            self.reload_match_agents()
        self.close()

    def _step_python(self):
        self._obs_recv_buffer = np.array(self.envs.gather(None, root=MPI.ROOT))

        if 'Unity' in self.type:
            # N sets of Roles
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    # role_actions = []
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    # try batching all role observations to the agent
                    # role_actions.append(self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate']))
                    # for o in role_obs:
                    #     role_actions.append(self.agents[agent_ix].get_action(o, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate']))
                    # env_actions.append(role_actions)
                    env_actions.append(self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate']))
                actions.append(env_actions)
        elif 'Particle' in self.type:
            # 1 set of Roles
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate'])
                    env_actions.append(role_actions)
                actions.append(env_actions)
        elif 'Gym' in self.type:
            actions = []
            role_ix = 0 # Single Role Environment
            role_name = self.env_specs['roles'][role_ix]
            agent_ix = self.role2agent[role_name]
            for role_obs in self._obs_recv_buffer:
                env_actions = []
                role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate'])
                env_actions.append(role_actions)
                actions.append(env_actions)

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        self.actions = np.array(actions)
        self.log("Shape Obs {} Acs {}".format(self._obs_recv_buffer.shape, self.actions.shape), verbose_level=2)
        self.log("Obs {} Acs {}".format(self._obs_recv_buffer, actions), verbose_level=3)
        # self.log("Step {}".format(self.step_count), verbose_level=2)

        if self.is_running:
            self.envs.scatter(actions, root=MPI.ROOT)
        else:
            self.envs.scatter([False] * self.num_envs, root=MPI.ROOT)

    def _step_numpy(self):
        '''
            For Numpy step, is required that
            - all agents observations are the same shape
            - all agents actions are the same shape
        '''
        self.envs.Gather(None, [self._obs_recv_buffer, MPI.DOUBLE], root=MPI.ROOT)
        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs

        if 'Unity' in self.type or 'Particle' in self.type:
            '''self._obs_recv_buffer receives data from many MPIEnv.py'''
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_actions = []
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    for o in role_obs:
                        role_actions.append(self.agents[agent_ix].get_action(o, self.step_count))
                    env_actions.append(role_actions)
                actions.append(env_actions)
            # actions = [ [ [self.agents[ix].get_action(o, self.step_count, self.learners_specs[ix]['evaluate']) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]
        elif 'Gym' in self.type:
            '''self._obs_recv_buffer receives data from many MPIEnv.py'''
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_actions = []
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    # for o in role_obs:
                    role_actions.append(self.agents[agent_ix].get_action(role_obs, self.step_count))
                    env_actions.append(role_actions)
                actions.append(env_actions)
            # actions = [ [ [self.agents[ix].get_action(o, self.step_count, self.learners_specs[ix]['evaluate']) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]
        elif 'RoboCup' in self.type:
            actions = [[agent.get_action(obs, self.step_count, self.device) for agent, obs in zip(self.agents, observations)] for observations in self._obs_recv_buffer]

        actions = np.array(actions)
        self.log("Obs {} Acs {}".format(self._obs_recv_buffer, self.actions), verbose_level=3)
        self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)

    def check_state(self):
        while self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=self.info):
            done_count = self.envs.recv(None, source=self.info.Get_source(), tag=Tags.trajectory_info)
            # self.log(f"Recv from Env {self.info.Get_source()} {done_count}")
            self.done_count += done_count
            self._time_to_load = True

        if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.close, status=self.info):
            # Close signal
            _ = self.meta.recv(None, source=self.info.Get_source(), tag=Tags.close)
            # used only to stop the whole session, for running profiling experiments..
            self.is_running = False

    def reload_match_agents(self, bypass_request=False):
        # if self.step_count % (self.episode_max_length * self.num_envs) == 0:
        if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents, status=self.info):
            '''In case a new match is received from MetaLearner'''
            self._receive_match(bypass_request=bypass_request)

        elif self._time_to_load and self.done_count > 0 and (self.done_count % self.episodic_load_rate == 0):
            '''No match available - reload current agents'''
            self.agents = self.load_agents(bypass_request=bypass_request)
            for a in self.agents:
                a.reset_noise()
            self._time_to_load = False

    def _receive_match(self, bypass_request=False):
        '''New match from the single MetaLearner'''
        self.role2learner_spec = self.meta.recv(None, source=0, tag=Tags.new_agents)
        self.log("Received Training Match {}".format(self.role2learner_spec), verbose_level=2)
        self._update_match_data(self.role2learner_spec, bypass_request=bypass_request)

    def _update_match_data(self, role2learner_spec, bypass_request=False):
        self.role2learner_id = {role:role2learner_spec[role]['id'] for role in self.env_specs['roles']}
        self.agents = self.load_agents(role2learner_spec, bypass_request=bypass_request)
        self.role2agent = self.get_role2agent_ix(self.agents) # for local usage
        # forward learner specs to all single environments
        for env_id in range(self.num_envs):
            self.envs.send(role2learner_spec, dest=env_id, tag=Tags.new_agents)

    def get_role2agent_ix(self, agents):
        '''Create Role->agent_id mapping for local usage'''
        self.role2agent = {}
        for role in self.env_specs['roles']:
            for ix, agent in enumerate(agents):
                if role == agent.role:
                    self.role2agent[role] = ix
                    break
        return self.role2agent

    def load_agents(self, role2learner_spec=None, bypass_request=False):
        if role2learner_spec is None:
            role2learner_spec = self.role2learner_spec

        _force_load = not hasattr(self, 'agents')
        agents = self.agents if hasattr(self, 'agents') else [None for i in range(len(self.env_specs['roles']))]
        for role, learner_spec in role2learner_spec.items():
            '''During runtime loops we only need to load the agents that are not being evaluated'''
            if _force_load or not learner_spec['evaluate']:

                if not bypass_request:
                    self.io.request_io(self._get_menv_specs(), learner_spec['load_path'], wait_for_access=True)
                learner_agents = Admin._load_agents(learner_spec['load_path'], device=self.device, load_latest=True)
                if not bypass_request:
                    self.io.done_io(self._get_menv_specs(), learner_spec['load_path'])

                for a in learner_agents:
                    '''Force Agent to use self.device'''
                    a.to_device(self.device)
                    agents[self.env_specs['roles'].index(a.role)] = a

        self.log("Loaded {}".format([str(agent) for agent in agents]), verbose_level=1)
        return agents

    def _receive_learner_spec(self, learner_ix):
        learner_spec = self.learners.recv(None, source=learner_ix, tag=Tags.specs) # blocking statement
        if learner_ix <= len(self.learners_specs) - 1:
            '''Replace if already there'''
            self.learners_specs[learner_ix] = learner_spec
        else:
            self.learners_specs.append(learner_spec)

    def _connect_learners(self):
        self.learners = MPI.COMM_WORLD.Accept(self.port) # Connect with Learners group
        # Get LearnersSpecs to load agents and start running
        self.learners_specs = []
        self.num_learners = self.configs['MetaLearner']['num_learners']
        for ix in range(self.num_learners):
            self._receive_learner_spec(ix)
        self.log("Got LearnerSpecs from {}".format([spec['id'] for spec in self.learners_specs]), verbose_level=1)

        # Cast LearnersSpecs to single envs for them to communicate with Learners
        self.envs.bcast(self.learners_specs, root=MPI.ROOT)
        envs_states = self.envs.gather(None, root=MPI.ROOT)

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIEnv.py'], maxprocs=self.num_envs)
        self.configs['MultiEnv'] = {}
        self.configs['MultiEnv']['id'] = self.id
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces) for Learners
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them
        self.log("Got {} EnvSpecs".format(len(envs_spec)), verbose_level=1)

    def _get_menv_specs(self):
        return {
            'type': 'MultiEnv',
            'id': self.id,
            'port': self.port,
            'env_specs': self.env_specs,
            'num_envs': self.num_envs if hasattr(self, 'num_envs') else None,
            'num_learners': self.num_learners if hasattr(self, 'num_learners') else None
        }

    def _connect_io_handler(self):
        self.io = get_io_stub(self.configs)

    def close(self):
        self.log("Started closing", verbose_level=2)
        for i in range(self.num_envs):
            self.envs.send(True, dest=i, tag=Tags.close)
        self.envs.Disconnect()
        self.log("Closed Envs", verbose_level=2)
        self.learners.Disconnect()
        MPI.Close_port(self.port) # close Learners port
        self.log("Closed Learners", verbose_level=2)
        self.meta.Disconnect()
        self.log("FULLY CLOSED", verbose_level=1)
        exit(0)

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['MultiEnv']:
            text = '{}\t{}'.format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<MultiEnv(id={}({}), episodes={} device={})>".format(self.id, self.num_envs, self.done_count, self.device)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        # self.log("LEARNER = Inter: {} / Intra: {}".format(self.learners.Is_inter(), self.learners.Is_intra()))


if __name__ == "__main__":
    try:
        menv = MPIMultiEnv()
    except Exception as e:
        msg = "<MultiEnv(id={}) error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
        terminate_process()
    finally:
        pass
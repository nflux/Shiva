import sys, time, traceback, subprocess, torch
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
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
        self.device = torch.device('cpu')

        # Open Port (for Learners)
        self.port = MPI.Open_port(MPI.INFO_NULL)

        self._launch_envs()

        # Handshake with Meta to send Environment Specs and my Port for Learners to connect
        self.meta.gather(self._get_menv_specs(), root=0)

        self._connect_learners()
        self._receive_match() # receive first match

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)
        self.run()

    def run(self):
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

        while True:
            time.sleep(self.configs['Admin']['time_sleep']['MultiEnv'])
            self._step_python()
            # self._step_numpy()
            if self.step_count % self.episode_max_length == 0:
                self.reload_match_agents() # temporarily training matches are fixed
                for a in self.agents:
                    a.reset_noise()

    def _step_python(self):
        self._obs_recv_buffer = self.envs.gather(None, root=MPI.ROOT)

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if 'Unity' in self.type:
            # N sets of Roles
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
        elif 'Particle' in self.type:
            # 1 set of Roles
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count)
                    env_actions.append(role_actions)
                actions.append(env_actions)
        elif 'Gym' in self.type:
            actions = []
            role_ix = 0 # Single Role Environment
            role_name = self.env_specs['roles'][role_ix]
            agent_ix = self.role2agent[role_name]
            for role_obs in self._obs_recv_buffer:
                env_actions = []
                role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count)
                env_actions.append(role_actions)
                actions.append(env_actions)

        self.actions = np.array(actions)
        self.log("Obs {} Acs {}".format(self._obs_recv_buffer, actions), verbose_level=3)
        self.log("Step {}".format(self.step_count), verbose_level=2)
        self.envs.scatter(actions, root=MPI.ROOT)

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
            actions = [[agent.get_action(obs, self.step_count,self.device) for agent, obs in zip(self.agents, observations)] for observations in self._obs_recv_buffer]

        actions = np.array(actions)
        self.log("Obs {} Acs {}".format(self._obs_recv_buffer, self.actions), verbose_level=3)
        self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)

    def reload_match_agents(self):
        if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents, status=self.info):
            pass
            '''If new match is received from MetaLearner'''
            # STATIC FOR NOW
            # self._receive_match()
        else:
            '''No match available - reload current agents'''
            self.agents = self.load_agents()

    def _receive_match(self):
        '''New match from the single MetaLearner'''
        self.role2learner_spec = self.meta.recv(None, source=0, tag=Tags.new_agents)
        self.log("Received Training Match {}".format(self.role2learner_spec), verbose_level=2)
        self._update_match_data(self.role2learner_spec)

    def _update_match_data(self, role2learner_spec):
        self.role2learner_id = {role:role2learner_spec[role]['id'] for role in self.env_specs['roles']}
        self.agents = self.load_agents(role2learner_spec)
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

    def load_agents(self, role2learner_spec=None):
        if role2learner_spec is None:
            role2learner_spec = self.role2learner_spec
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        _ = self.io.recv(None, source=0, tag=Tags.io_menv_request)
        agents = self.agents if hasattr(self, 'agents') else [None for i in range(len(self.env_specs['roles']))]
        for role, learner_spec in role2learner_spec.items():
            '''Need to load ONLY the agents that are not being evaluated'''
            if not learner_spec['evaluate']:
                learner_agents = Admin._load_agents(learner_spec['load_path'])
                for a in learner_agents:
                    agents[self.env_specs['roles'].index(a.role)] = a
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
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

    '''Abstraction of the IO request'''
    def _io_permission(self, function_call, *args, **kwargs):
        self.io_tag = Tags.io_menv_request
        self.io.send(True, dest=0, tag=self.io_tag)
        _ = self.io.recv(None, source=0, tag=self.io_tag)
        r = function_call(*args, **kwargs)
        self.io.send(True, dest=0, tag=self.io_tag)
        return r

    def _connect_io_handler(self):
        self.io = MPI.COMM_WORLD.Connect(self.menvs_io_port, MPI.INFO_NULL)
        self.log('IOHandler connected', verbose_level=1)

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['MultiEnv']:
            text = '{}\t{}'.format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<MultiEnv(id={}, num_envs={})>".format(self.id, self.num_envs)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        # self.log("LEARNER = Inter: {} / Intra: {}".format(self.learners.Is_inter(), self.learners.Is_intra()))


if __name__ == "__main__":
    try:
        menv = MPIMultiEnv()
    except Exception as e:
        print("MultiEnv error:", traceback.format_exc())
    finally:
        terminate_process()

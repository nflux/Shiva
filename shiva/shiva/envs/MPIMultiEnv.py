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
    num_envs = False
    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEnv, self).__init__(self.configs)
        self._connect_io_handler()
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        self.device = torch.device('cpu')
        # Open Port for Learners
        self.port = MPI.Open_port(MPI.INFO_NULL)
        # self.log("Open port {}".format(self.port))

        '''Set self attrs from Config'''
        self.num_learners = self.configs['MetaLearner']['num_learners']

        self._launch_envs()
        self.meta.gather(self._get_menv_specs(), root=0) # checkin with Meta
        self._connect_learners()

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)
        self.run()

    def run(self):
        self.step_count = 0
        self.done_count = 0
        self.log(self.env_specs)
        self.saving = [True] * self.num_learners

        '''
            Check if we can do a numpy step instead of python list
            If obs dimensions for all roles are the same, then we can do MPI
        '''

        if 'Unity' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0] ), dtype=np.float64)
        elif 'Gym' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], self.env_specs['observation_space'] ), dtype=np.float64)
        elif 'RoboCup' in self.type:
            self._obs_recv_buffer = np.empty((self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space']), dtype=np.float64)

        while True:
            self._step_python()
            # self._step_numpy()

            '''TODO: after X amount of steps or dones, load new agents'''
            if self.step_count % self.episode_max_length == 0:
                self.agents = self.load_agents()
                for a in self.agents:
                    a.reset_noise()

    def _step_python(self):
        self._obs_recv_buffer = self.envs.gather(None, root=MPI.ROOT)
        # self.log("Obs Shape {}".format(np.array([self._obs_recv_buffer]).shape))

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if 'Unity' in self.type:
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
            role_ix = 0
            role_name = self.env_specs['roles'][role_ix]
            agent_ix = self.role2agent[role_name]
            for role_obs in self._obs_recv_buffer:
                env_actions = []
                role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count)
                env_actions.append(role_actions)
                actions.append(env_actions)
        self.actions = np.array(actions)
        self.log("Obs {} Acs {}".format(self._obs_recv_buffer, actions))
        self.envs.scatter(actions, root=MPI.ROOT)

    def _step_numpy(self):
        '''
            For Numpy step, is required that
            - all agents observations are the same shape
            - all agents actions are the same shape
        '''
        self.envs.Gather(None, [self._obs_recv_buffer, MPI.DOUBLE], root=MPI.ROOT)
        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        # self.log("{}\n{}\n{}\n{}".format([str(a) for a in self.agents], self.role2agent, self.envs_role2learner, self._obs_recv_buffer))
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
            self.log(self._obs_recv_buffer)
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
            # self.log("The actions shape {}".format(actions))

        actions = np.array(actions)
        # self.log("Obs {} Acs {}".format(self._obs_recv_buffer, self.actions))
        self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces) for Learners
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them
        self.log("Got {} EnvSpecs".format(len(envs_spec)))

    def load_agents(self):
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        _ = self.io.recv(None, source=0, tag=Tags.io_menv_request)
        agents = self.agents
        for learner_spec in self.learners_specs:
            '''Need to load ONLY the agents that are not being evaluated'''
            if not learner_spec['evaluate']:
                learner_agents = Admin._load_agents(learner_spec['load_path'])
                for a in learner_agents:
                    agents[self.env_specs['roles'].index(a.role)] = a
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        # self.log("Loading {}".format([str(agent) for agent in agents]))
        return agents

    def load_agents_init(self):
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        _ = self.io.recv(None, source=0, tag=Tags.io_menv_request)
        agents = [i for i in range(len(self.env_specs['roles']))]
        for learner_spec in self.learners_specs:
            learner_agents = Admin._load_agents(learner_spec['load_path'])
            for a in learner_agents:
                agents[self.env_specs['roles'].index(a.role)] = a
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        # self.log("Loading Init {}".format([str(agent) for agent in agents]))
        return agents
        # return flat_1d_list([Admin._load_agents(learner_spec['load_path']) for learner_spec in self.learners_specs])

    def _receive_learner_spec(self, learner_ix):
        learner_spec = self.learners.recv(None, source=learner_ix, tag=Tags.specs) # blocking statement
        if learner_ix <= len(self.learners_specs) - 1:
            self.learners_specs[learner_ix] = learner_spec
        else:
            self.learners_specs.append(learner_spec)
        self.log("Received Learner<{}>".format(learner_spec['id']))

    def _connect_learners(self):
        self.learners = MPI.COMM_WORLD.Accept(self.port) # Wait until check in learners, create comm
        # Get LearnersSpecs to load agents and start running
        self.learners_specs = []
        self.log("Expecting {} Learners".format(self.num_learners))
        for ix in range(self.num_learners):
            self._receive_learner_spec(ix)
        self.log("Got {} Learners: {}".format(len(self.learners_specs), self.learners_specs))

        self.agents = self.load_agents_init()

        '''Create Role->AgentIX mapping'''
        self.role2agent = {}
        for role in self.env_specs['roles']:
            for ix, agent in enumerate(self.agents):
                if role == agent.role:
                    self.role2agent[role] = ix
                    break

        # Cast LearnersSpecs to single envs for them to communicate with Learners
        self.envs.bcast(self.learners_specs, root=MPI.ROOT)
        '''Need match making process here'''
        self.envs_role2learner =  self.get_envs_role2learner()
        self.envs.scatter(self.envs_role2learner, root=MPI.ROOT)
        '''Get signal from single env that they have connected with Learner'''
        envs_states = self.envs.gather(None, root=MPI.ROOT)
        # self.log(envs_status)

    def get_envs_role2learner(self):
        '''Assuming every Environment will have the same Role->Learner mapping - TODO: match making process for random/unique combinations'''
        self.role2learner = {}
        for role in self.env_specs['roles']:
            for learner_spec in self.learners_specs:
                if role in learner_spec['roles']:
                    self.role2learner[role] = learner_spec['id']
                    break
        self.envs_role2learner = [self.role2learner for i in range(self.num_envs)]
        return self.envs_role2learner

    def _get_menv_specs(self):
        return {
            'type': 'MultiEnv',
            'id': self.id,
            'port': self.port,
            'env_specs': self.env_specs,
            'num_envs': self.num_envs,
            'num_learners': self.num_learners
        }

    def _connect_io_handler(self):
        # self.log('Sending IO Connectiong Request')
        self.io = MPI.COMM_WORLD.Connect(self.menvs_io_port, MPI.INFO_NULL)
        self.log('IOHandler connected')

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = '{}\t{}'.format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<MultiEnv(id={}, num_envs={})>".format(self.id, self.num_envs)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.log("LEARNER = Inter: {} / Intra: {}".format(self.learners.Is_inter(), self.learners.Is_intra()))


if __name__ == "__main__":
    try:
        menv = MPIMultiEnv()
    except Exception as e:
        print("MultiEnv error:", traceback.format_exc())
    finally:
        terminate_process()

import sys, time, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.envs.Environment import Environment
from shiva.helpers.misc import terminate_process

class MPIMultiEnv(Environment):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEnv, self).__init__(self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        # Open Port for Learners
        self.port = MPI.Open_port(MPI.INFO_NULL)
        self.log("Open port {}".format(self.port))

        '''Set self attrs from Config'''
        self.num_learners = self.configs['MetaLearner']['num_learners']
        self.num_envs = self.num_instances # number of childrens

        self._launch_envs()
        self.meta.gather(self._get_menv_specs(), root=0) # checkin with Meta
        self._connect_learners()

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)

        self.run()

    def run(self):
        self.step_count = 0
        self.log(self.env_specs)
        info = MPI.Status()

        ''' Assuming that
            - all agents have the same observation shape, if they don't then we have a multidimensional problem for MPI
            - agents_instances are in equal amount for all agents
        '''
        if 'Unity' in self.env_specs['type']:
            self._obs_recv_buffer = np.zeros((self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0]), dtype=np.float32)
        else:
            self._obs_recv_buffer = np.zeros((self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space']), dtype=np.float32)

        while True:
            # self._step_python_list()
            self._step_numpy()

            if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents, status=info):
                learner_id = info.Get_source()
                learner_spec = self.learners.recv(None, source=learner_id, tag=Tags.new_agents)
                '''Assuming 1 Agent per Learner'''
                self.agents[learner_id] = Admin._load_agents(learner_spec['load_path'])[0]

        self.close()
    
    def _step_numpy(self):
        # self.log("Getting stuck before gather")
        self.envs.Gather(None, [self._obs_recv_buffer, MPI.FLOAT], root=MPI.ROOT)
        # self.log("Obs Shape {}".format(self._obs_recv_buffer.dtype))

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if self.env_specs['num_instances_per_env'] > 1:
            '''Unity case!!'''
            '''self._obs_recv_buffer receives data from many MPIEnv.py'''
            actions = [[[self.agents[ix].get_action(o, self.step_count) for o in obs] for ix, obs in enumerate(observations) ] for observations in self._obs_recv_buffer]
        else:
            # implement for Gym and Robocup
            # self.log("Obs {}".format(self._obs_recv_buffer[0]))
            actions = [[agent.get_action(obs, self.step_count) for agent, obs in zip(self.agents, observations)] for observations in self._obs_recv_buffer]
            # self.log("Acs {}".format(actions))
        
        # self.log("Acs Shape 1 {}".format(actions))
        actions = np.array(actions, dtype=np.float32)
        # self.log("Actions {} Obs {}".format(actions, self._obs_recv_buffer))
        # self.log("{} {}".format(self.actions[0][0][0][0], self.actions[0][1][0][0]))

        # self.log("Acs Shape 2 {}".format(self.actions))
        self.envs.Scatter([actions, MPI.FLOAT], None, root=MPI.ROOT)

    def _step_python_list(self):
        '''We could optimize this gather/scatter ops using numpys'''
        self.observations = self.envs.gather(None, root=MPI.ROOT)
        self.log("Obs {}".format(self.observations))

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if self.env_specs['num_instances_per_env'] > 1:
            '''Unity case!!'''
            actions = [[self.agents[ix].get_action(o, self.step_count) for o in obs] for ix, obs in enumerate(self.observations)]
        else:
            actions = self.agents[0].get_action(self.observations, self.step_count)  # assuming one agent for all obs
        self.log("Acs {}".format(actions))
        self.envs.scatter(actions, root=MPI.ROOT)

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/envs/MPIEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces)
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them

    def _connect_learners(self):
        self.learners = MPI.COMM_WORLD.Accept(self.port) # Wait until check in learners, create comm
        # Get LearnersSpecs to load agents and start running
        self.learners_specs = []
        self.log("Expecting {} learners".format(self.num_learners))
        for i in range(self.num_learners):
            learner_data = self.learners.recv(None, source=i, tag=Tags.specs)
            self.learners_specs.append(learner_data)
            self.log("Received Learner {}".format(learner_data['id']))

        '''
            TODO
                - Assuming one learner above
                - load centralized/decentralized agents using the config
        '''
        self.agents = [ Admin._load_agents(learner_spec['load_path'])[0] for learner_spec in self.learners_specs ]

        # Cast LearnersSpecs to single envs for them to communicate with Learners
        self.envs.bcast(self.learners_specs, root=MPI.ROOT)
        # Get signal that they have communicated with Learner
        envs_states = self.envs.gather(None, root=MPI.ROOT)
        # self.log(envs_status)

    def _get_menv_specs(self):
        return {
            'type': 'MultiEnv',
            'id': self.id,
            'port': self.port,
            'env_specs': self.env_specs,
            'num_envs': self.num_instances
        }

    def close(self):
        self.learners.Unpublish_name()
        self.learners.Close_port()
        self.envs.Disconnect()
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()
        MPI.COMM_WORLD.Abort()

    def log(self, msg, to_print=False):
        text = 'Menv {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.log("LEARNER = Inter: {} / Intra: {}".format(self.learners.Is_inter(), self.learners.Is_intra()))


if __name__ == "__main__":
    try:
        menv = MPIMultiEnv()
    except Exception as e:
        print("MultiEnv error:", traceback.format_exc(), flush=True)
    finally:
        terminate_process()
import sys, time, traceback, subprocess
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
        self.io = MPI.COMM_WORLD.Connect(self.menvs_io_port, MPI.INFO_NULL)
        self.info = MPI.Status()
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        # Open Port for Learners
        self.port = MPI.Open_port(MPI.INFO_NULL)
        #self.log("Open port {}".format(self.port))

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
        self.saving = [True] * self.num_learners

        ''' Assuming that
            - all agents have the same observation shape, if they don't then we have a multidimensional problem for MPI
            - agents_instances are in equal amount for all agents
        '''

        if 'Unity' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0] ), dtype=np.float64)
        elif 'Gym' in self.type:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], self.env_specs['observation_space'] ), dtype=np.float64)
        elif 'RoboCup' in self.type:
            self._obs_recv_buffer = np.empty((self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space']), dtype=np.float64)

        while True:
            time.sleep(0.001)
            # self._step_python_list()
            self._step_numpy()

            #if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents, status=info):
                # self.log("THERE ARE NEW AGENTS TO LOAD")
                #learner_id = info.Get_source()
                #learner_spec = self.learners.recv(None, source=learner_id, tag=Tags.new_agents)
                # self.log("These are the learner specs {}".format(learner_spec))
                #'''Assuming 1 Agent per Learner'''
                # if self.update_nums[learner_id] != learner_spec['update_num']:
                # self.log("About to load {}".format(learner_id))
                #self.saving[learner_id] = learner_spec['load']
                #if not self.saving[learner_id]:
                    #self.saving[learner_id] = True
                    #self.agents[learner_id] = Admin._load_agents(learner_spec['load_path'])[0]
                    # self.log("Agent Loaded From Learner {}".format(learner_id))
                    #self.learners.send(True, dest=learner_id, tag=Tags.save_agents)
                    # self.update_nums[learner_id] = learner_spec['update_num']

            if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents, status=self.info):
                #self._io_load_agents()
                 learner_id = self.info.Get_source()
                 learner_spec = self.learners.recv(None, source=learner_id, tag=Tags.new_agents)
                 #'''Assuming 1 Agent per Learner'''
                 self.io.send(True, dest=0, tag=Tags.io_menv_request)
                 _ = self.io.recv(None, source = 0, tag=Tags.io_menv_request)
                 self.agents[learner_id] = Admin._load_agents(learner_spec['load_path'])[0]
                 self.io.send(True, dest=0, tag=Tags.io_menv_request)
                 self.log("Got LearnerSpecs<{}> and loaded Agent at Episode {} / Step {}".format(learner_id, self.agents[learner_id].done_count, self.agents[learner_id].step_count))
        # self.close()

    def _step_numpy(self):
        self.envs.Gather(None, [self._obs_recv_buffer, MPI.DOUBLE], root=MPI.ROOT)

        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs

        if 'Unity' in self.type:
            '''self._obs_recv_buffer receives data from many MPIEnv.py'''
            actions = [[[self.agents[ix].get_action(o, self.step_count, self.learners_specs[ix]['evaluate']) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]
            actions = np.array(actions)
            self.envs.scatter(actions, root=MPI.ROOT)
        elif 'Gym' in self.type:
            # Gym
            # same?
            actions = [[[self.agents[ix].get_action(o, self.step_count, self.learners_specs[ix]['evaluate']) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]
            actions = np.array(actions)
            self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)
        elif 'RoboCup' in self.type:
            actions = [[agent.get_action(obs, self.step_count) for agent, obs in zip(self.agents, observations)] for observations in self._obs_recv_buffer]
            actions = np.array(actions)
            # self.log("The actions shape {}".format(actions))
            self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)


        # self.log("Obs {} Acs {}".format(self._obs_recv_buffer, self.actions))

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIImitationEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces)
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them

    def _connect_learners(self):
        self.learners = MPI.COMM_WORLD.Accept(self.port) # Wait until check in learners, create comm
        # Get LearnersSpecs to load agents and start running
        self.learners_specs = []
        #self.log("Expecting {} learners".format(self.num_learners))
        for i in range(self.num_learners):
            '''Learner IDs are inserted in order :)'''
            learner_spec = self.learners.recv(None, source=i, tag=Tags.specs)
            self.learners_specs.append(learner_spec)
            #self.log("Received Learner {}".format(learner_spec['id']))

        '''
            TODO
                - Assuming one learner above
                - load centralized/decentralized agents using the config
        '''
        #self.log("Got all Learners Specs\n\t{}".format(self.learners_specs))
        '''Assuming 1 Agent per Learner, we could break it with a star operation'''
        self.io.send(True, dest=0 , tag=Tags.io_menv_request)
        _ = self.io.recv(None, source = 0, tag=Tags.io_menv_request)
        self.agents = [ Admin._load_agents(learner_spec['load_path'])[0] for learner_spec in self.learners_specs ]
        self.io.send(True, dest=0, tag=Tags.io_menv_request)
        #self.agents = [ Admin._load_agents(learner_spec['load_path'])[0] for learner_spec in self.learners_specs ]

        # Cast LearnersSpecs to single envs for them to communicate with Learners
        self.envs.bcast(self.learners_specs, root=MPI.ROOT)
        # Get signal from single env that they have connected with Learner
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

    def _io_load_agents(self):
        learner_id = self.info.Get_source()
        learner_spec = self.learners.recv(None, source=learner_id, tag=Tags.new_agents)
        '''Assuming 1 Agent per Learner'''
        self.io.send(learner_spec, dest=0, tag=Tags.io_checkpoint_load)
        self.agents[learner_id] = self.io.recv(None, source=MPI.ANY_SOURCE,tag=Tags.io_checkpoint_load)

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg):
        text = 'Menv {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, self.configs['Admin']['print_debug'])

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

import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.envs.Environment import Environment

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

        # Grab configs
        self.num_learners = 1 # assuming 1 Learner
        self.num_envs = self.num_instances

        self._launch_envs()
        self.meta.gather(self._get_menv_specs(), root=0) # checkin with Meta
        self._connect_learners()

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)

        self.run()

    def run(self):
        self.step_count = 0
        self.log(self.env_specs)
        while True:
            '''We could optimize this gather/scatter ops using numpys'''
            observations = self.envs.gather(None, root=MPI.ROOT)
            # self.log("Obs {}".format(observations))
            self.step_count += len(observations)

            if self.env_specs['num_instances_per_env'] > 1:
                '''Unity case!!'''
                actions = [self.agents[0].get_action(obs, self.step_count) for obs in observations]
            else:
                actions = [agent.get_action(obs, self.step_count) for agent, obs in zip(self.agents, observations)]
                # actions = [ in self.agents[0].get_action(observations, self.step_count) # assuming one agent for all obs
            # self.log("Acs {}".format(actions))
            self.envs.scatter(actions, root=MPI.ROOT)

            if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.new_agents):
                '''Assuming 1 Learner, need to grab the source rank to load the appropiate agent'''
                learner_spec = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.new_agents)  # block statement
                self.agents = Admin._load_agents(learner_spec['load_path'])
                self.log("Loaded Agent at Episode {}".format(self.agents[0].done_count))
                
        self.close()

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
        self.agents = Admin._load_agents(self.learners_specs[0]['load_path'])

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
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'Menv {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.log("LEARNER = Inter: {} / Intra: {}".format(self.learners.Is_inter(), self.learners.Is_intra()))


if __name__ == "__main__":
    MPIMultiEnv()
import time, os, sys
from mpi4py import MPI

from shiva.metalearners.CommMultiLearnerMetaLearnerServer import get_meta_stub
from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub
from shiva.helpers.launch_servers_helper import start_menv_server
import shiva.envs.CommEnvironment as comm_env

from shiva.helpers.config_handler import load_class

class CommMultiEnvironment():
    def __init__(self, id):
        self.id = id
        self.address = ':'.join(['localhost', '50002'])

    def launch(self, meta_address):
        self.meta_stub = get_meta_stub(meta_address)

        self.debug("gRPC Request Meta for Configs")
        self.configs = self.meta_stub.get_configs()
        self.debug("gRPC Received Config from Meta")
        {setattr(self, k, v) for k, v in self.configs['Environment'].items()}

        # initiate server
        self.comm_menv_server, self.menv_tags = start_menv_server(self.address, maxprocs=1)
        # self.debug("MPI send Configs to MultiEnvServer")
        req = self.comm_menv_server.isend(self.configs, 0, self.menv_tags.configs)
        req.Wait()

        # initiate individual envs
        self.env_p = []
        self.debug("Ready to instantiate {} environment/s of {}".format(self.num_instances, self.configs['Environment']['type']))

        self.init_envs()
        # MultiEnv receives EnvSpecs from single Envs
        self.check_in_envs()

        self.debug("Total of {} environments checked in".format(self.checks))

        self.meta_stub.send_menv_specs(self._get_menv_specs()) # sends handshake to Meta that includes the MultiEnvSpecs

        self.debug("Sent MultiEnvSpecs to Meta")
        self.debug("Ready to check in Learners & Agents")

        self.check_in_learners()
        # self.envs_comm.bcast(self.learners_specs, root=MPI.ROOT) # send learners specs to all environments thru MPI?

        self.envs_comm.bcast([True], root=MPI.ROOT) # all environments can now run!

    def init_envs(self):
        file = comm_env.__file__
        self.envs_comm = MPI.COMM_SELF.Spawn(sys.executable, args=[file, '-pa', self.address], maxprocs=self.num_instances)
        self.envs_comm.bcast(self.configs, root=MPI.ROOT)

    def check_in_learners(self):
        # receive learners & agents to be loaded
        specs = None
        self.learners_specs = []
        self.learners_stubs = {}
        self.checks = 0
        while self.checks < self.env_specs['num_agents']:
            specs = self.comm_menv_server.recv(None, 0, self.menv_tags.learner_specs)
            self.learners_specs.append(specs)
            # self.learners_stubs[specs['id']] = get_learner_stub(specs['address']) # is this needed?
            self.checks += specs['num_agents']
        self.debug("A total of {} Learners checked in".format(len(self.learners_specs)))

    def check_in_envs(self):
        env_specs = None
        self.checks = 0
        while self.checks < self.num_instances:
            env_specs = self.comm_menv_server.recv(None, 0, self.menv_tags.env_specs) # MPI
            self.checks += 1
            # self.debug("Environment {} checked in".format(env_specs['id']))
        self.env_specs = env_specs
        return None

    def debug(self, msg):
        print("PID {} MultiEnv\t\t{}".format(os.getpid(), msg))

    def _get_menv_specs(self):
        return {
            'id': self.id,
            'env_specs': self.env_specs,
            'num_envs': self.num_instances,
            'address': self.address
        }
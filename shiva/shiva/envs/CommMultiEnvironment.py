import time, os
import torch.multiprocessing as mp

from shiva.metalearners.CommMultiLearnerMetaLearnerServer import get_meta_stub
from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub
from shiva.helpers.launch_servers_helper import start_menv_server
from shiva.envs.CommEnvironment import create_comm_env

from shiva.helpers.config_handler import load_class

class CommMultiEnvironment():
    def __init__(self, id):
        self.id = id
        self.address = ':'.join(['localhost', '50002'])
        self.num_instances = 2

    def launch(self, meta_address):
        self.meta_stub = get_meta_stub(meta_address)

        # self.debug("gRPC Request Meta for Configs")
        self.configs = self.meta_stub.get_configs()
        self.debug("gRPC Received Config from Meta")
        {setattr(self, k, v) for k, v in self.configs['Environment'].items()}

        # initiate server
        self.comm_menv_server, self.menv_tags = start_menv_server(self.address, maxprocs=1)
        # self.debug("MPI send Configs to MultiEnvServer")
        self.comm_menv_server.send(self.configs, 0, self.menv_tags.configs)

        # initiate individual envs
        self.env_p = []
        env_cls = load_class('shiva.envs', self.configs['Environment']['type'])  # all envs with same config
        self.debug("Ready to instantiate {} environment/s of {}".format(self.num_instances, self.configs['Environment']['type']))
        menv_stub = get_menv_stub(self.address)
        for ix in range(self.num_instances):
            env_id = ix
            p = mp.Process(target=create_comm_env, args=(env_cls, env_id, self.configs, menv_stub))
            p.start()
            time.sleep(1) # give some time for each individual environment to connect
            self.env_p.append(p)

        # MultiEnv receives EnvSpecs from single Envs
        self.check_in_envs()

        self.debug("Total of {} environments checked in".format(self.checks))

        self.meta_stub.send_menv_specs(self._get_menv_specs()) # sends handshake to Meta that includes the MultiEnvSpecs

        self.debug("Sent MultiEnvSpecs to Meta")
        self.debug("Ready to check in Learners & Agents")

        self.check_in_learners()

    def check_in_learners(self):
        # receive learners & agents to be loaded
        specs = None
        self.learners_specs = []
        self.learners_stubs = {}
        self.checks = 0
        while self.checks < self.env_specs['num_agents']:
            specs = self.comm_menv_server.recv(None, 0, self.menv_tags.learner_specs)
            self.learners_specs.append(specs)
            self.learners_stubs[specs['id']] = get_learner_stub(specs['address']) # is this needed?
            self.checks += specs['num_agents']
        self.debug("A total of {} Learners checked in".format(len(self.learners_specs)))

    def check_in_envs(self):
        env_specs = None
        self.checks = 0
        while self.checks < self.num_instances:
            env_specs = self.comm_menv_server.recv(None, 0, self.menv_tags.env_specs)
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
import time, os
import torch.multiprocessing as mp
from mpi4py import MPI

from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

from shiva.helpers.launch_servers_helper import start_meta_server, check_port
from shiva.helpers.launch_components_helper import start_menv, start_learner
from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub

class CommMultiLearnerMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(CommMultiLearnerMetaLearner, self).__init__(configs)
        self.configs = configs
        self.port = '50001'
        check_port(self.port)
        self.address = ':'.join(['localhost', '50001'])

        # initiate server
        self.comm_meta_server, self.meta_tags = start_meta_server(self.address, maxprocs=1)
        # self.debug('MPI Send Configs to MetaServer')
        req = self.comm_meta_server.isend(self.configs, dest=0, tag=self.meta_tags.configs)
        req.Wait()

        '''
            Assuming 1 MultiEnv for now!
            Need to think how the Config would look for all scenarios of decentralized/centralized learners
        '''
        # initiate multienv processes
        self.num_menvs = 1
        for menv_id in range(self.num_menvs):
            self.debug("Starts MultiEnv # {}".format(menv_id))
            self.comm_menv = start_menv(menv_id, self.address)

        # received menv specs & create stubs as they come in
        self.check_in_menvs()
        # self.comm_menv.bcast(self.configs, root=MPI.ROOT)

        # initiate learner processes
        '''
            Problem when multiple Learners, they initialize their server on the same address :/
        '''
        self.num_learners = 1
        for learner_id in range(self.num_learners):
            # self.debug("Starts Learner # {}".format(learner_id))
            self.comm_learner = start_learner(learner_id, self.address)
        # receive learners specs & create stubs as they come in
        self.check_in_learners()
        # self.comm_learner.bcast(self.configs, root=MPI.ROOT)

        '''
            Here we need to decide how to distribute the Environment Specs across the Learners
            
            For now, Assuming 1 Learner and 1 MultiEnv
        '''
        for learner_id, learner_stub in self.learners_stubs.items():
            menv_spec = self.menv_specs[0]
            learner_stub.send_menv_specs(menv_spec)
            self.debug("Sent MultiEnvSpecs {} to Learner {}".format(menv_spec['id'], learner_id))

        self.run()

    def check_in_menvs(self):
        self.menv_stubs = {}
        self.menv_specs = []
        menv_specs = None
        self.checks = 0
        while self.checks < self.num_menvs:
            menv_specs = self.comm_meta_server.recv(None, 0, self.meta_tags.menv_specs)
            self.checks += 1
            # self.debug("MultiEnv {} checked in".format(menv_specs['id']))
            self.menv_specs.append(menv_specs)
            self.menv_stubs[menv_specs['id']] = get_menv_stub(menv_specs['address'])
        self.debug("Total of {} MultiEnvs checked in".format(self.checks))
        return None

    def check_in_learners(self):
        self.learners_stubs = {}
        self.learner_specs = []
        learner_specs = None
        self.checks = 0
        while self.checks < self.num_learners:
            learner_specs = self.comm_meta_server.recv(None, 0, self.meta_tags.learner_specs)
            self.checks += 1
            # self.debug("Learner {} checked in".format(learner_specs['id']))
            self.learner_specs.append(learner_specs)
            self.learners_stubs[learner_specs['id']] = get_learner_stub(learner_specs['address'])
        self.debug("Total of {} Learners checked in".format(self.checks))
        return None

    def run(self):
        while True:
            pass
            # receive TrainingMetrics + NewAgents from learnersl
            #   if found, send NewAgents + EvaluationConfig to Evaluation

            # check evaluation metrics
            #   if found, do PBT and distribute evolution configs to learners


    def debug(self, msg):
        print('PID {} Meta\t\t\t{}'.format(os.getpid(), msg))

    def close(self):
        print('Cleaning up!')
        # self.comm_meta_server.Barrier()
        # # stop counter process
        # rank = self.comm_meta_server.Get_rank()
        # if rank == 0:
        #     self.comm_meta_server.send(None, source=0, tag=self.meta_tags.close)
        # self.comm_meta_server.Disconnect()
        # self.comm_meta_server.Free()
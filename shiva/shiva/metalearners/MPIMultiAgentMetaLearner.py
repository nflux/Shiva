import sys
import logging
import numpy as np
import torch
from mpi4py import MPI

from shiva.core.admin import logger
from shiva.metalearners.MetaLearner import MetaLearner

class MPIMultiAgentMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIMultiAgentMetaLearner, self).__init__(configs, profile=False)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.configs = configs
        self.launch()

    def launch(self):
        self._launch_menvs()
        self._launch_learners()
        self.run()

    def run(self):
        while True:
            learner_specs = self.learners.gather(None, root=MPI.ROOT)
            self.log("Got Learners metrics {}".format(learner_specs))

    def _launch_menvs(self):
        self.menvs = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(len(menvs_specs), len(menvs_specs[0].keys())))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/learners/MPIMultiAgentLearner.py'], maxprocs=self.num_learners)
        self.learners.bcast(self.configs, root=MPI.ROOT)
        learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.log("Got {}".format(learners_specs))

    def log(self, msg, to_print=False):
        text = "Meta\t\t{}".format(msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def close(self):
        self.menvs.Disconnect()
        self.learners.Disconnect()

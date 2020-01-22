import sys
import logging
import numpy as np
from mpi4py import MPI

from shiva.metalearners.MetaLearner import MetaLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

class MPIMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self.launch()

    def launch(self):
        self._launch_menvs()
        self._launch_learners()
        self.run()

    def run(self):
        while True:
            learner_specs = self.learners.gather(None, root=MPI.ROOT)
            # self.debug("Got Learners metrics {}".format(learner_specs))

    def _launch_menvs(self):
        self.menvs = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.debug("Got MenvsSpecs {}".format(menvs_specs))
        self.debug("Got total of {} MultiEnvsSpecs with {} keys".format(len(menvs_specs), len(menvs_specs[0].keys())))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        self.learners.bcast(self.configs, root=MPI.ROOT)
        learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.debug("Got {}".format(learners_specs))

    def debug(self, msg, to_print=False):
        text = "Meta\t\t{}".format(msg)
        logging.debug(text)
        if to_print or self.configs['Admin']['debug']:
            print(text)

    def close(self):
        self.menvs.Disconnect()

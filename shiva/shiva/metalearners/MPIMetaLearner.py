import sys
import logging
import numpy as np
from mpi4py import MPI

from shiva.core.admin import logger
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_config_file_2_dict

class MPIMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        # self._process_config()
        self.launch()

    def launch(self):
        self._launch_menvs()
        self._launch_learners()
        self.run()

    def run(self):
        while True:
            learner_specs = self.learners.gather(None, root=MPI.ROOT)
            self.log("Got Learners metrics {}".format(learner_specs), )

    def _launch_menvs(self):
        self.menvs = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(len(menvs_specs), len(menvs_specs[0].keys())))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        '''
            Check correct assignment of self.learners_map with self.configs['MultiEnv']['env_specs']['agents_group']
         '''
        # self._preprocess_config_assignment()
        # for ix in self.num_learners:
        #
        #     self.learners.send()
        self.learners.bcast(self.configs, root=MPI.ROOT)
        learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.log("Got {}".format(learners_specs))

    def _preprocess_config_assignment(self):
        self.learner_configs = {}
        '''Check that the Learners assignment with the environment Group Names are correct'''
        for agent_group in self.configs['MultiEnv']['env_specs']['agents_group']:
            if agent_group in set(self.learners_map.keys()):
                pass
            else:
                assert "Config error - Agent Group {} was not found on the [MetaLearner] learners_map attribute".format(agent_group)

            '''Load the config for each learner'''
            if agent_group not in self.learner_configs:
                self.learner_configs[agent_group] = load_config_file_2_dict()
        confir_dir = self.learners_map[agent_group]

    def _process_config(self):
        '''Do some preprocessing before spreading the config'''
        self.num_learners = len(set(self.learners_map.keys()))
        self.configs['MetaLearner']['num_learners'] = self.num_learners

    def log(self, msg, to_print=False):
        text = "Meta\t\t{}".format(msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def close(self):
        '''Send message to childrens'''
        #
        self.menvs.Disconnect()
        self.learners.Disconnect()

import sys
import logging
import numpy as np
from mpi4py import MPI

from shiva.core.admin import logger
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_config_file_2_dict, merge_dicts
from shiva.helpers.misc import terminate_process

class MPIMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self._preprocess_config()
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
        self.menv_configs = self._get_menv_configs()
        self.menvs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.menv_configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(len(menvs_specs), len(menvs_specs[0].keys())))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners_configs = self._get_learners_configs()
        self.learners = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        # self.log("Scattering {}".format(self.learners_configs))
        self.learners.scatter(self.learners_configs, root=MPI.ROOT)
        learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.log("Got {} LearnerSpecs".format(len(learners_specs)))

    def _get_menv_configs(self):
        self.menv_configs = []
        if hasattr(self, 'learners_map'):
            ''' Create a friendly Learners map for the MultiEnv '''
            self.menv_configs = self.configs.copy()
        elif 'Learner' in self.configs:
            self.menv_configs = self.configs.copy()
        return self.menv_configs

    def _get_learners_configs(self):
        '''
            Check that the Learners assignment with the environment Role Names are correct
            This will only run if the learners_map is set
        '''
        self.learner_configs = []
        if hasattr(self, 'learners_map'):
            '''First check that all Agents Roles are assigned to a Learner'''
            for ix, roles in enumerate(self.configs['MultiEnv'][0]['env_specs']['roles']):
                if roles in set(self.learners_map.keys()):
                    pass
                else:
                    assert "Agent Roles {} is not being assigned to any Learner\nUse the 'learners_map' attribute on the [MetaLearner] section".format(roles)
            '''Do some preprocessing before spreading the config'''
            # load each one of the configs and keeping same order
            self.learners_configs = []
            for config_path, learner_roles in self.learners_map.items():
                learner_config = load_config_file_2_dict(config_path)
                learner_config = merge_dicts(self.configs, learner_config)
                learner_config['Learner']['roles'] = learner_roles
                self.learners_configs.append(learner_config)
            # for config_path in list(self.learners_map.keys()):
            #     learner_config = load_config_file_2_dict(config_path)
            #     self.learners_configs.append(merge_dicts(self.configs, learner_config))
        elif 'Learner' in self.configs:
            self.learners_configs = [self.configs.copy() for _ in range(self.num_learners)]
        else:
            assert "Error processing Learners Configs"
        return self.learners_configs

    def _preprocess_config(self):
        if hasattr(self, 'learners_map'):
            # calculate number of learners using the learners_map dict
            self.num_learners = len(set(self.learners_map.keys()))
            self.configs['MetaLearner']['num_learners'] = self.num_learners
        else:
            # num_learners is explicitely in the config
            pass

    def log(self, msg, to_print=False):
        text = "Meta\t\t{}".format(msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def close(self):
        '''Send message to childrens'''
        #
        self.menvs.Disconnect()
        self.learners.Disconnect()

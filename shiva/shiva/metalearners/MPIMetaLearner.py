import sys
import logging
import numpy as np
from mpi4py import MPI

from shiva.core.admin import logger
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.utils.Tags import Tags
from shiva.helpers.config_handler import load_config_file_2_dict, merge_dicts
from shiva.helpers.misc import terminate_process
from shiva.utils.Tags import Tags

class MPIMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self._preprocess_config()
        self.launch()

    def launch(self):
        self._launch_io_handler()
        self._launch_menvs()
        self._launch_learners()
        self.run()

    def run(self):
        while True:
            learner_specs = self.learners.gather(None, root=MPI.ROOT)
            self.log("Got Learners metrics {}".format(learner_specs))

    def _launch_io_handler(self):
        self.io = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/helpers/io_handler.py'], maxprocs=1)
        self.io.send(self.configs, dest=0, tag=Tags.configs)
        self.io_specs = self.io.recv(None, source=0, tag=Tags.io_config)
        if 'Learner' in self.configs:
            self.configs['Environment']['menvs_io_port'] = self.io_specs['menvs_port']
            self.configs['Learner']['learners_io_port'] = self.io_specs['learners_port']
            # self.configs['Evaluation']['evals_io_port'] = self.io_specs['evals_port']
        else:
            '''With the learners_map, self.configs['Learner'] dict doesn't exist yet - will try some approaches here'''
            self.configs['IOHandler'] = {}
            self.configs['Environment']['menvs_io_port'] = self.io_specs['menvs_port']
            self.configs['IOHandler']['menvs_io_port'] = self.io_specs['menvs_port']
            self.configs['IOHandler']['learners_io_port'] = self.io_specs['learners_port']
            # self.configs['IOHandler']['evals_io_port'] = self.io_specs['evals_port']

    def _launch_menvs(self):
        self.menvs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(str(len(menvs_specs)), str(len(menvs_specs[0].keys()))))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners_configs = self._get_learners_configs()
        self.learners = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        self.learners.scatter(self.learners_configs, root=MPI.ROOT)
        learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.log("Got {} LearnerSpecs".format(len(learners_specs)))

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
                learner_config['Learner']['learners_io_port'] = self.configs['IOHandler']['learners_io_port']
                self.learners_configs.append(learner_config)
        elif 'Learner' in self.configs:
            self.learners_configs = [self.configs.copy() for _ in range(self.num_learners)]
        else:
            assert "Error processing Learners Configs"
        return self.learners_configs

    def _preprocess_config(self):
        if not hasattr(self, 'pbt'):
            self.configs['MetaLearner']['pbt'] = False
        if hasattr(self, 'learners_map'):
            # calculate number of learners using the learners_map dict
            self.num_learners = len(set(self.learners_map.keys()))
            self.configs['MetaLearner']['num_learners'] = self.num_learners
        else:
            # num_learners is explicitely in the config
            pass

    def log(self, msg, to_print=False):
        text = "{}\t\t{}".format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<Meta>"

    def close(self):
        '''Send message to childrens'''
        self.menvs.Disconnect()
        self.learners.Disconnect()

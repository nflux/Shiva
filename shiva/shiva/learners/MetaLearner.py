import numpy as np
from abc import abstractmethod

from shiva.core.admin import Admin, logger
from shiva.learners.Learner import Learner

from typing import List, Dict, Tuple, Any, Union

class MetaLearner(object):
    def __init__(self, configs: Dict[str, Any], profile: bool=True):
        """
        This class is the root of all Shiva processes. The Meta Learner is used to interface between the Learning pipeline and the Evaluations pipeline when PBT is used.

        Args:
            configs (Dict[str, Any]): Config to be run
            profile (bool): this is used for the non-distributed run (could be deprecated)
        """
        {setattr(self, k, v) for k,v in configs['MetaLearner'].items()}
        self.configs = configs
        self.manual_seed = np.random.randint(10000) if not hasattr(self, 'manual_seed') else self.manual_seed
        if profile:
            Admin.add_meta_profile(self, self.get_folder_name())

    @abstractmethod
    def evolve(self) -> None:
        """
        Performs evolution procedures. It uses the rankings received by the evaluations in order to send a evolution config to the Learner.
        This function is executed only when a Learner has requested an evolution config.

        Returns:
            None
        """
        raise NotImplementedError("Method to be implemented by subclass")

    def create_learner(self) -> Learner:
        """
        Since Shiva is currently developed under a distributed architecture, this function is not being used but instead `_launch_learners` where the Learners processes are spawned.
        This function should be used for a non-distributed architecture.

        Returns:
            Learner
        """
        raise NotImplementedError("Method to be implemented by subclass")

    def get_folder_name(self) -> str:
        """
        Format to be used for the folder name where we are gonna save all checkpoints.

        Returns:
            str: folder name for the run
        """
        return '-'.join([self.config['Algorithm']['type'], self.config['Environment']['env_name']])

    def log(self, msg, verbose_level=-1):
        """
        Logging function. Uses python logger and can optionally output to terminal depending on the config `['Admin']['print_debug']`

        Args:
            msg: Message to be logged
            verbose_level: verbose level used for the given message. Defaults to -1.

        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['MetaLearner']:
            text = "{}\t\t{}".format(str(self), msg)
            logger.info(text, to_print=self.configs['Admin']['print_debug'])

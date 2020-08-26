import numpy as np
from shiva.core.admin import Admin, logger


class MetaLearner(object):
    def __init__(self, configs, profile=True):
        """
        This class is the root of all Shiva processes. The Meta Learner is used to interface between the Learning pipeline and the Evaluations pipeline when PBT is used.

        Args:
            configs (Dict[str, Any]): Config to be run
            profile (bool): this is used for the non-distributed run (could be deprecated)
        """
        {setattr(self, k, v) for k,v in configs['MetaLearner'].items()}
        self.configs = configs
        self.manual_seed = np.random.randint(10000) if not hasattr(self, 'manual_seed') else self.manual_seed
        self.learnerCount = 0
        if profile:
            Admin.add_meta_profile(self, self.get_folder_name())

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self):
        pass

    def genetic_crossover(self):
        pass

    def evolve(self):
        """
        Performs evolution procedures. It uses the rankings received by the evaluations in order to send a evolution config to the Learner.
        This function is executed only when a Learner has requested an evolution config.

        Returns:
            None
        """
        pass

    def evaluate(self):
        pass

    def record_metrics(self):
        pass

    def create_learner(self):
        """
        Since Shiva is currently developed under a distributed architecture, this function is not being used but instead `_launch_learners` where the Learners processes are spawned.
        This function should be used for a non-distributed architecture.

        Returns:
            Learner
        """
        raise NotImplemented

    def get_id(self):
        return self.get_new_learner_id()

    def get_new_learner_id(self):
        id = self.learnerCount
        self.learnerCount += 1
        return id

    def get_folder_name(self):
        """
        Format to be used for the folder name where we are gonna save all checkpoints.

        Returns:
            str: folder name for the run
        """
        try:
            folder_name = '-'.join([self.config['Algorithm']['type'], self.config['Environment']['env_name']])
        except:
            folder_name = '-'.join([self.config['Algorithm']['type1'], self.config['Environment']['env_name']])
        return folder_name

    def save(self):
        Admin.save(self)

    def log(self, msg, to_print=False, verbose_level=-1):
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
            logger.info(text, to_print=to_print or self.configs['Admin']['print_debug'])

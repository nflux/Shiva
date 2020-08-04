from shiva.core.admin import logger
from abc import abstractmethod, ABC


class Evaluation(ABC):
    """ Evaluation Abstract

    This branch of Shiva is for ranking agents for Population Based Training

    """
    def __init__(self, configs: 'whole config passed'):
        if 'Evaluation' in configs:
            {setattr(self, k, v) for k, v in configs['Evaluation'].items()}
        else:
            {setattr(self, k, v) for k, v in configs.items()}

    @abstractmethod
    def evaluate_agents(self):
        """ Starts evaluation process"""

    @abstractmethod
    def _create_eval_envs(self):
        """This implementation is specific to each environment type"""

    @abstractmethod
    def _start_evals(self):
        """This implementation is specific to each environment type"""

    @abstractmethod
    def rank_agents(self, validation_scores):
        """ Ranks the agents using various metrics

            Could be rewards, or metrics specific to the environment.
        """

    def log(self, msg, verbose_level=-1):
        """If verbose_level is not given, by default will log"""
        if verbose_level <= self.configs['Admin']['log_verbosity']['Evaluation']:
            text = "{}\t{}".format(str(self), msg)
            logger.info(text, self.configs['Admin']['print_debug'])

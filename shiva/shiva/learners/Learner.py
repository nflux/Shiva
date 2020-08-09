import torch
from abc import abstractmethod

from shiva.core.admin import Admin, logger
from shiva.core.TimeProfiler import TimeProfiler
from shiva.helpers.config_handler import load_class

from typing import List, Dict, Tuple, Any, Union

class Learner(object):
    
    def __init__(self, learner_id: int, config: Dict[str, Dict[str, Any]]):
        """
        Abstract class for the Learners.

        Args:
            learner_id (int): Unique ID for the Learner
            config (Dict): config dictioanry to be used for the run
        """
        self.num_agents_created = 0
        self.ep_count = 0
        self.step_count = 0
        self.evolution_count = 0
        self.n_evolution_requests = 0
        self.checkpoints_made = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        Admin.init(self.configs)
        Admin.add_learner_profile(self, function_only=True)
        self.profiler = TimeProfiler(self.configs, Admin.get_learner_url_summary(self))

    def collect_metrics(self):
        """
        High level function to collect metrics for all agents that this Learner owns and then send them to process to be plotted in tensorboard.

        Returns:
            None
        """
        if hasattr(self, 'agent'):
            if type(self.agent) == list:
                for agent in self.agent:
                    metrics = self.get_metrics(agent.id) + self.alg.get_metrics(agent.id)
                    self._process_metrics(agent.id, metrics)
            else:
                metrics = self.get_metrics(self.agent.id) + self.alg.get_metrics(self.agent.id)
                self._process_metrics(self.agent.id, metrics)
        elif hasattr(self, 'agents'):
            for agent in self.agents:
                metrics = self.get_metrics(agent.id) + self.alg.get_metrics(agent.id)
                self._process_metrics(agent.id, metrics)
        else:
            assert False, "Learner attribute 'agent' or 'agents' was not found..."

    def _process_metrics(self, agent_id: int, metrics: List[Union[List[Tuple[str, float, int]], Tuple[str, float, int]]]):
        """

        Args:
            agent_id (int): Agent ID for who we are processing the metrics
            metrics (List): list of tuple metrics OR list of list of tuple metrics

        Returns:
            None
        """
        for m in metrics:
            # self.log("Agent {}, Metric {}".format(agent_id, m))
            if type(m) == list:
                '''Is a list of tuple metrics'''
                for metric_tuple in m:
                    self._add_summary_writer(agent_id, metric_tuple)
            elif type(m) == tuple:
                '''One single tuple metric'''
                self._add_summary_writer(agent_id, m)

    def _add_summary_writer(self, agent_id: int, metric_tuple: Union[Tuple[str, float], Tuple[str, float, int]]):
        """
        Directly interacts with ShivaAdmin to plot to tensorboard

        Args:
            agent_id (int): Agent ID
            metric_tuple (Tuple): tuple containing (metric name, y value, x value)

        Returns:
            None
        """
        if len(metric_tuple) == 3:
            metric_name, y_value, x_value = metric_tuple
        elif len(metric_tuple) == 2:
            metric_name, y_value = metric_tuple
            x_value = self.done_count
        Admin.add_summary_writer(self, agent_id, metric_name, y_value, x_value)

    def checkpoint(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def update(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def step(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def create_environment(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def get_agents(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def get_algorithm(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def launch(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def exploitation(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def exploration(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def save(self):
        Admin.save(self)

    def load(self, attrs):
        for key in attrs:
            setattr(self, key, attrs[key])

    def close(self, *args, **kwargs):
        raise NotImplementedError("Method to be implemented by subclass")

    def get_id(self):
        return self.get_new_agent_id()

    def log(self, msg, to_print=False, verbose_level=-1):
        """
        Logging function. Uses python logger and can optionally output to terminal depending on the config `['Admin']['print_debug']`

        Args:
            msg: Message to be logged
            verbose_level: verbose level used for the given message. Defaults to -1.

        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['Learner']:
            text = '{}\t\t{}'.format(str(self), msg)
            logger.info(text, to_print=self.configs['Admin']['print_debug'])

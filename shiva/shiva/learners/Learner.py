import torch
from shiva.core.admin import Admin, logger
from shiva.core.TimeProfiler import TimeProfiler
from shiva.helpers.config_handler import load_class

class Learner(object):
    num_agents_created = 0
    ep_count = 0
    step_count = 0
    evolution_count = 0
    n_evolution_requests = 0
    checkpoints_made = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, learner_id, config, port=None):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        self.port = port
        Admin.init(self.configs)
        Admin.add_learner_profile(self, function_only=True)
        self.profiler = TimeProfiler(self.configs, Admin.get_learner_url_summary(self))

    def collect_metrics(self, episodic=False):
        '''
            This works for Single Agent Learner
            For Multi Agent Learner we need to implement the else statement
        '''
        if hasattr(self, 'agent'):
            if type(self.agent) == list:
                '''Multi Agent Metrics'''
                for agent in self.agent:
                    self._collect_metrics(agent.id, episodic)
            else:
                self._collect_metrics(self.agent.id, episodic)
        elif hasattr(self, 'agents'):
            if type(self.agents) == list:
                '''Multi Agent Metrics'''
                for agent in self.agents:
                    self._collect_metrics(agent.id, episodic)
            else:
                self._collect_metrics(self.agents[0].id, episodic)
        else:
            assert False, "Learner attribute 'agent' or 'agents' was not found..."

    def _collect_metrics(self, agent_id, episodic):
        try:
            metrics = self.get_metrics(episodic, agent_id)
        except:
            metrics = self.get_metrics(episodic)

        if not self.evaluate:
            try:
                metrics += self.alg.get_metrics(episodic, agent_id)
            except:
                metrics += self.alg.get_metrics(episodic)
        self._process_metrics(agent_id, metrics)

    def _process_metrics(self, agent_id, metrics):
        for m in metrics:
            # self.log("Agent {}, Metric {}".format(agent_id, m))
            if type(m) == list:
                '''Is a list of tuple metrics'''
                for metric_tuple in m:
                    self._add_summary_writer(agent_id, metric_tuple)
            elif type(m) == tuple:
                '''One single tuple metric'''
                self._add_summary_writer(agent_id, m)

    def _add_summary_writer(self, agent_id, metric_tuple):
        if len(metric_tuple) == 3:
            metric_name, y_value, x_value = metric_tuple
        elif len(metric_tuple) == 2:
            metric_name, y_value = metric_tuple
            x_value = self.done_count
        Admin.add_summary_writer(self, agent_id, metric_name, y_value, x_value)

    def checkpoint(self):
        raise NotImplemented

    def update(self):
        raise NotImplemented

    def step(self):
        raise NotImplemented

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def get_agents(self):
        raise NotImplemented

    def get_algorithm(self):
        raise NotImplemented

    def launch(self):
        raise NotImplemented

    def save(self):
        Admin.save(self)

    def load(self, attrs):
        for key in attrs:
            setattr(self, key, attrs[key])

    def close(self):
        self.env.close()

    def get_id(self):
        return self.get_new_agent_id()

    def log(self, msg, verbose_level=-1):
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

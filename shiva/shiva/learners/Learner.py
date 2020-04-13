import torch

from shiva.core.admin import Admin, logger
from shiva.helpers.config_handler import load_class

class Learner(object):
    
    def __init__(self, learner_id, config, port=None):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        self.port = port
        self.agentCount = 0
        self.ep_count = 0
        self.step_count = 0
        self.checkpoints_made = 0
        self.totalReward = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getstate__(self):
        d = dict(self.__dict__)
        attributes_to_ignore = ['env', 'envs', 'eval', 'queue', 'queues', 'miniBuffer']
        return [] # Ezequiel: Unsure who added this but needs documentation - maybe because we don't need anything when save/loading the Learner?
        for t in d:
            if t not in attributes_to_ignore:
                # print(t)
                pass
        for a in attributes_to_ignore:
            try:
                del d[a]
            except:
                pass
        return d

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
        if hasattr(self, 'MULTI_ENV_FLAG'):
            try:
                metrics = self.get_metrics(episodic, agent_id)
            except:
                metrics = self.get_metrics(episodic)

            if not self.evaluate:
                try:
                    metrics += self.alg.get_metrics(episodic, agent_id)
                except:
                    metrics += self.alg.get_metrics(episodic)
            self._add_summary_writer(agent_id, metrics)
        else:
            assert False, "Testing if this if statement is unnecessary"
            # metrics = self.alg.get_metrics(episodic) + self.env.get_metrics(episodic)
            # if not episodic:
            #     for metric_name, y_val in metrics:
            #         Admin.add_summary_writer(self, agent_id, metric_name, y_val, self.env.step_count)
            # else:
            #     for metric_name, y_val in metrics:
            #         Admin.add_summary_writer(self, agent_id, metric_name, y_val, self.env.done_count)

    def _add_summary_writer(self, agent_id, metrics):
        for m in metrics:
            # self.log("Agent {}, Metric {}".format(agent_id, m))
            if type(m) == list:
                '''Is a list of set metrics'''
                for metric_name, y_val in m:
                    Admin.add_summary_writer(self, agent_id, metric_name, y_val, self.done_count)
            elif type(m) == tuple:
                '''One single set metric'''
                metric_name, y_val = m
                Admin.add_summary_writer(self, agent_id, metric_name, y_val, self.done_count)

    def checkpoint(self):
        assert hasattr(self, 'save_checkpoint_episodes'), "Learner needs 'save_checkpoint_episodes' attribute in config - put 0 if don't want to save checkpoints"
        if self.save_checkpoint_episodes > 0:
            t = self.save_checkpoint_episodes * self.checkpoints_made
            if self.env.done_count > t:
                print("%% Saving checkpoint at episode {} %%".format(self.env.done_count))
                Admin.checkpoint(self)
                self.checkpoints_made += 1

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

    def get_new_agent_id(self):
        id = self.agentCount
        self.agentCount +=1
        return id

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Learner']:
            text = '{}\t\t{}'.format(str(self), msg)
            logger.info(text, to_print=to_print or self.configs['Admin']['print_debug'])

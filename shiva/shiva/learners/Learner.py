import torch

from shiva.core.admin import Admin
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
        attributes_to_ignore = ['env', 'envs', 'eval', 'queue', 'queues','miniBuffer']
        return []
        for t in d:
            if t not in attributes_to_ignore:
                print(t)
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
                print("MultiAgents metrics are not yet implemented! 1")
                return
            else:
                self._collect_metrics('agent', episodic)
        elif hasattr(self, 'agents'):
            if type(self.agents) == list:
                print("MultiAgents metrics are not yet implemented! 2")
                return
            else:
                self._collect_metrics('agents', episodic)
        else:
            assert False, "Learner attribute 'agent' or 'agents' was not found..."

    def _collect_metrics(self, attr, episodic):
        # if not multi environment
        # if multi environment

        if hasattr(self, 'MULTI_ENV_FLAG'):

            metrics = self.alg.get_metrics(episodic) + self.get_metrics(episodic)

            if not episodic:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, getattr(self, attr), metric_name, y_val, self.step_count)
            else:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, getattr(self, attr), metric_name, y_val, self.ep_count)
        else:

            metrics = self.alg.get_metrics(episodic) + self.env.get_metrics(episodic)

            if not episodic:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, getattr(self, attr), metric_name, y_val, self.env.step_count)
            else:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, getattr(self, attr), metric_name, y_val, self.env.done_count)



    def checkpoint(self):
        assert hasattr(self, 'save_checkpoint_episodes'), "Learner needs 'save_checkpoint_episodes' attribute in config - put 0 if don't want to save checkpoints"
        if self.save_checkpoint_episodes > 0:
            t = self.save_checkpoint_episodes * self.checkpoints_made
            if self.env.done_count > t:
                print("%% Saving checkpoint at episode {} %%".format(self.env.done_count))
                Admin.checkpoint(self)
                self.checkpoints_made += 1

    def update(self):
        assert 'Not implemented'
        pass

    def step(self):
        assert 'Not implemented'
        pass

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def get_agents(self):
        assert 'Not implemented'
        pass

    def get_algorithm(self):
        assert 'Not implemented'
        pass

    def launch(self):
        assert 'Not implemented'
        pass

    def save(self):
        Admin.save(self)

    def load(self, attrs):
        for key in attrs:
            setattr(self, key, attrs[key])

    def close(self):
        pass

    def get_id(self):
        return self.get_new_agent_id()

    def get_new_agent_id(self):
        id = self.agentCount
        self.agentCount +=1
        return id

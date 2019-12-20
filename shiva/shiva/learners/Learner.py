from shiva.core.admin import Admin
from shiva.helpers.config_handler import load_class

class Learner(object):
    
    def __init__(self, learner_id, config):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        self.agentCount = 0
        self.ep_count = 0
        self.step_count = 0
        

    def __getstate__(self):
        d = dict(self.__dict__)
        try:
            del d['eval']
            del d['env']
        except KeyError:
            del d['envs']
        return d

    def collect_metrics(self, episodic=False):
        '''
            This works for Single Agent Learner
            For Multi Agent Learner we need to implemenet the else statement
        '''
        if hasattr(self, 'agent') and type(self.agent) is not list:
            metrics = self.alg.get_metrics(episodic) + self.env.get_metrics(episodic)
            if not episodic:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, self.agent, metric_name, y_val, self.env.step_count)
            else:
                for metric_name, y_val in metrics:
                    Admin.add_summary_writer(self, self.agent, metric_name, y_val, self.env.done_count)
        else:
            assert False, "The Learner attribute 'agent' was not found. Either name the attribute 'agent' or could be that MultiAgent Metrics are not yet supported."
    
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

    def get_id(self):
        id = self.agentCount
        self.agentCount +=1
        return id
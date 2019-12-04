from settings import shiva

class Learner(object):
    
    def __init__(self, learner_id, config):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        self.agentCount = 0
        self.ep_count = 0

    def __getstate__(self):
        d = dict(self.__dict__)
        try:
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
                    shiva.add_summary_writer(self, self.agent, metric_name, y_val, self.step_count)
            else:
                for metric_name, y_val in metrics:
                    shiva.add_summary_writer(self, self.agent, metric_name, y_val, self.ep_count)
        else:
            assert False, "The Learner attribute 'agent' was not found. Either name the attribute 'agent' or could be that MultiAgent Metrics are not yet supported."
    
    def update(self):
        pass

    def step(self):
        pass

    def create_env(self, alg):
        pass

    def get_agents(self):
        pass

    def get_algorithm(self):
        pass

    def launch(self):
        pass

    def save(self):
        shiva.save(self)

    def load(self, attrs):
        for key in attrs:
            setattr(self, key, attrs[key])

    def get_id(self):
        id = self.agentCount
        self.agentCount +=1
        return id
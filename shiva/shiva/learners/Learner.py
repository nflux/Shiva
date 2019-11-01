from settings import shiva

class Learner(object):
    
    def __init__(self, learner_id, config):
        {setattr(self, k, v) for k,v in config['Learner'].items()}
        self.configs = config
        self.id = learner_id
        self.agentCount = 0

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
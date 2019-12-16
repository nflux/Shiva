from __main__ import shiva

class MetaLearner(object):
    def __init__(self, config):
        {setattr(self, k, v) for k,v in config['MetaLearner'].items()}
        self.config = config
        self.learnerCount = 0
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        folder_name = '-'.join([config['Algorithm']['type'], config['Environment']['env_name']])
        shiva.add_meta_profile(self, folder_name)

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self, hp, algorithms):
        pass

    def genetic_crossover(self,agents, elite_agents):
        pass

    def evolve(self, new_agents, new_hp):
        pass

    def evaluate(self, learners: list):
        pass

    def record_metrics(self):
        pass

    def create_learner(self, learner_id, config):
        pass

    def get_id(self):
        id = self.learnerCount
        self.learnerCount +=1
        return id

    def save(self):
        shiva.save(self)
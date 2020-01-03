from shiva.core.admin import Admin

class MetaLearner(object):
    def __init__(self, configs):
        {setattr(self, k, v) for k,v in configs['MetaLearner'].items()}
        self.configs = configs
        self.episodes = configs['Learner']['episodes']
        self.learnerCount = 0
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        try:
            folder_name = '-'.join([configs['Algorithm']['type'], configs['Environment']['env_name']])
        except:
            folder_name = '-'.join([configs['Algorithm']['type1'], configs['Environment']['env_name']])
        Admin.add_meta_profile(self, folder_name)

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self, hp, algorithms):
        pass

    def genetic_crossover(self,agents, elite_agents):
        pass

    def evolve(self, new_agents, new_hp):
        pass

    def evaluate(self):
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
        Admin.save(self)
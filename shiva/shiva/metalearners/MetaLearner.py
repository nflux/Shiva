from shiva.core.admin import Admin

class MetaLearner(object):
    def __init__(self, config, profile=True):
        {setattr(self, k, v) for k,v in config['MetaLearner'].items()}
        self.config = config
        self.episodes = config['Learner']['episodes']
        self.learnerCount = 0
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        if profile:
            Admin.add_meta_profile(self, self.get_folder_name())

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
        return self.get_new_learner_id()

    def get_new_learner_id(self):
        id = self.learnerCount
        self.learnerCount += 1
        return id

    def get_folder_name(self):
        try:
            folder_name = '-'.join([self.config['Algorithm']['type'], self.config['Environment']['env_name']])
        except:
            folder_name = '-'.join([self.config['Algorithm']['type1'], self.config['Environment']['env_name']])
        return folder_name

    def save(self):
        Admin.save(self)
from shiva.core.admin import Admin
from shiva.helpers.config_handler import load_class

class DummyMetaLearner:
    def __init__(self, configs):
        {setattr(self, k, v) for k,v in configs['MetaLearner'].items()}
        self.episodes = configs['Learner']['episodes']
        self.learnerCount = 0
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        self.configs = configs
        self.run()

    def run(self):

        if self.start_mode == self.EVAL_MODE:
            pass

        elif self.start_mode == self.PROD_MODE:

            self.learner = self.create_learner()

            self.learner.launch()

            self.learner.run()

            self.learner.close()

        print('bye')

    def create_learner(self):
        learner = load_class('shiva.learners', self.configs['Learner']['type'])
        if hasattr(self, 'start_port'):
            return learner(self.get_id(), self.configs, self.start_port)
        else:
            return learner(self.get_id(), self.configs)
    
    def get_id(self):
        id = self.learnerCount
        self.learnerCount += 1
        return id
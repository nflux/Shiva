from shiva.core.admin import Admin
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

class SingleAgentMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(SingleAgentMetaLearner, self).__init__(configs)
        self.configs = configs
        self.learnerCount = 0
        self.run()

    def run(self):

        if self.start_mode == self.EVAL_MODE:
            pass
            # self.eval_env = []
            # # Load Learners to be passed to the Evaluation
            # self.learners = [ shiva._load_learner(load_path) for load_path in self.configs[0]['Evaluation']['load_path'] ]

            # self.configs[0]['Evaluation']['learners'] = self.learners
            # # Create Evaluation class
            # self.eval_env.append(Evaluation.initialize_evaluation(self.configs[0]['Evaluation']))

            # self.eval_env[0].evaluate_agents()


        elif self.start_mode == self.PROD_MODE:

            self.learner = self.create_learner()

            Admin.add_learner_profile(self.learner)

            self.learner.launch()
            
            Admin.update_agents_profile(self.learner)

            self.learner.run()

            self.save()

        print('bye')

    def create_learner(self):
        learner = load_class('shiva.learners', self.configs['Learner']['type'])
        return learner(self.get_id(), self.configs)

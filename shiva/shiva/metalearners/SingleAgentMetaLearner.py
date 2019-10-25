# this is a version of a meta learner that will take a file path to the configuration files
from settings import shiva
from .MetaLearner import MetaLearner
from learners.SingleAgentDQNLearner import SingleAgentDQNLearner
from learners.SingleAgentDDPGLearner import SingleAgentDDPGLearner
import helpers.misc as misc
import learners

class SingleAgentMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(SingleAgentMetaLearner,self).__init__(configs)
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

            # agents, environments, algorithm, data, configs for a single agent learner
            #agents, environments, algorithm, data, config
            self.learner = self.create_learner()

            # self.learner = self.create_learner(self.agent, self.eval_env, self.algorithm, self.buffer, self.learner_config)
            shiva.add_learner_profile(self.learner)

            # initialize the learner instances
            self.learner.launch()
            shiva.update_agents_profile(self.learner)
            
            # Runs the learner for a number of episodes given by the config
            self.learner.run()

            # save
            self.save()
    
    def create_learner(self):
        learner = getattr(learners, self.configs['Learner']['type'])
        return learner(self.get_id(), self.configs)


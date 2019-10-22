# this is a version of a meta learner that will take a file path to the configuration files
from settings import shiva
from .MetaLearner import MetaLearner

class SingleAgentMetaLearner(MetaLearner):
    def __init__(self,
                learners : "list of learner objects but in this case there's only one but there could be more",
                algorithms : "list of algorithm name strings",
                eval_env : "the name of the evaluation environment; from config file",
                agents : "list of agents, in this case there's only one",
                elite_agents : "list of elite agent objects from evaluation environment",
                optimize_env_hp : "boolean for whether or not we are optimizing hyperparameters",
                optimize_learner_hp : "boolean to optimize learner hyperparameters",
                evolution : "boolean for whether are not we will use evolution",
                configs: "list of all config dictionaries"
        ):
        super(SingleAgentMetaLearner,self).__init__(learners, 
                                                    algorithms, 
                                                    eval_env, 
                                                    agents, 
                                                    elite_agents, 
                                                    optimize_env_hp, 
                                                    optimize_learner_hp, 
                                                    evolution,
                                                    configs
        )
        if self.mode == self.EVAL_MODE:
            
            self.eval_env = []
            # Load Learners to be passed to the Evaluation
            self.learners = [ shiva._load_learner(load_path) for load_path in self.configs[0]['Evaluation']['load_path'] ]

            self.configs[0]['Evaluation']['learners'] = self.learners
            # Create Evaluation class
            self.eval_env.append(Evaluation.initialize_evaluation(self.configs[0]['Evaluation']))
            
            self.eval_env[0].evaluate_agents()
            

        elif self.mode == self.PROD_MODE:

            # agents, environments, algorithm, data, configs for a single agent learner
            self.learners.append(Learner.SingleAgentLearner(self.id_generator(), [], [], self.algorithms, [], configs[0]))
            shiva.add_learner_profile(self.learners[0])
            # initialize the learner instances
            self.learners[0].launch()
            
            shiva.update_agents_profile(self.learners[0])
            
            # Rus the learner for a number of episodes given by the config
            self.learners[0].run()

            # save
            self.save()
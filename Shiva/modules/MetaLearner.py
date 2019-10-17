from settings import shiva
import Learner, Algorithm, Agent
import Evaluation

# Maybe add a function to dictate the meta learner based on the configs
# needs to keep track everytime its used so that it always gives a unique id

def initialize_meta(config):
    if config[0]['MetaLearner']['type'] == 'Single':
        return SingleAgentMetaLearner([],
                                    config[0]['Algorithm'], # 
                                    config[0]['MetaLearner']['eval_env'],  # the evaluation environment
                                    [],  # this would have to be a list of agent objects
                                    [],  # this would have to be a list of elite agents objects
                                    config[0]['MetaLearner']['optimize_env_hp'],
                                    config[0]['MetaLearner']['optimize_learner_hp'],
                                    config[0]['MetaLearner']['evolution'],
                                    config,
        )


class AbstractMetaLearner(object):
    def __init__(self,
                learners : list, 
                algorithms : list, 
                eval_env : str, 
                agents : list, 
                elite_agents : list, 
                optimize_env_hp : bool, 
                optimize_learner_hp : bool, 
                evolution : bool,
                configs : list):
        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self.evolution = evolution
        self.learnerCount = 0
        self.configs = configs
        
        env_name = self.configs[0]['Environment']['env_type'] + '-' + self.configs[0]['Environment']['environment']
        shiva.add_meta_profile(self, env_name)
        
        self.PROD_MODE, self.EVAL_MODE = 'production', 'evaluation'
        self.mode = self.configs[0]['MetaLearner']['start_mode']

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

    def id_generator(self):
        id = self.learnerCount
        self.learnerCount +=1
        return id

    def save(self):
        shiva.save(self)

# this is a version of a meta learner that will take a file path to the configuration files
class SingleAgentMetaLearner(AbstractMetaLearner):
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
            # Load Learners and Agents given in @load_path
            _loaded_learners = [shiva._load_learner(learner_url) for learner_url in self.configs[0]['Evaluation']['load_path']]
            # tweak config to pass to Evaluation class
            self.configs[0]['Evaluation']['learners'] = _loaded_learners
            # Instance Evaluation class
            self.eval_env.append(Evaluation.initialize_evaluation(self.configs[0]['Evaluation']))
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
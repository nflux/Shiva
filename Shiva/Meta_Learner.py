from Learner import Learner
from Validation import Validation

class AbstractMetaLearner():

    def __init__(self, 
                learners : list, 
                algorithms : list, 
                eval_env : str, 
                agents : list, 
                elite_agents : list, 
                optimize_env_hp : bool, 
                optimize_learner_hp : bool, 
                evolution : bool):

        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self.evolution = evolution

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

# this is a version of a meta learner that will take a file path to the configuration files
class MetaLearner(AbstractMetaLearner):


    def __init__(self, path):

        validation = Validation(path)

        if True:

            self.algorithms = validation.algorithms

            # list of learner objects
            learners = []

            # Either this is still Here might be where multiprocessing will begin

            # here we will go through each config and initialize the learners
            for learner, environment, algorithm in zip(validation.learners,validation.environments, validation.algorithms):
                learners.append(Learner([], environment, algorithm, [], learner))

            # store list of learner objects in metalearner
            self.learners = learners


            for learner in self.learners: 
                learner.launch() 

        else:
            # in this case validation.message will have something bad
            print(validation.message)
            # end the program somehow

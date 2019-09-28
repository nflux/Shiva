import Learner, Validation

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
        # there might be different algorithms that will tune the hyperparameters

        pass
        #return new_hp
    
    # these woud breed
    def genetic_crossover(self,agents, elite_agents):
        pass

        #return new_agents

    def evolve(self, new_agents, new_hp):

        pass

    def evaluate(self, learners):
        pass

    def record_metrics(self):
        pass

# this is a version of a meta learner that will take a file path to the configuration files
class MetaLearner(AbstractMetaLearner):


    #

    def __init__(self, path):

        validation = Validation.validate(path)           

        if validation.success():

            self.learners = validation.learners
            # here we will go through each config and initialize the learners
            for learner in self.learners:
                pass

        else:
            # in this case validation.message will have something bad
            print(validation.message)
            # end the program somehow

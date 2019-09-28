import Learner, Validation
import os

class AbstractMetaLearner():

    def __init__(self, path):
        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self. evolution = evolution

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

        self.algorithms
        pass

    def evaluate(self, learners):
        pass

    def record_metrics(self):
        pass

# this is a version of a meta learner that will take a file path to the configuration files
class MetaLearner(AbstractMetaLearner):


    #self, learners, algorithms, eval_env, agents, elite_agents, optimize_env_hp=False, optimize_learner_hp=False, evolution=False

    def __init__(self, path):

        validation = Validation.validate(path)

        if validation.success():
            
            # here we will go through each config and initialize the learners
            
            pass
        else:
            # in this case validation.message will have something bad
            print(validation.message)
            # end the program somehow

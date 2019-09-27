import Learner, Validation
import os

class AbstractMetaLearner():

    # So I'm thinking I can make reading the configuration file something that happens inside of the ini
    def __init__(self, config):

        # Passing the config file through validation should produce all the hyperparameters that need to be implemented in 
        # different algorithms

        for c in config:
            Validation.validate(c)

        #self, learners, algorithms, eval_env, agents, elite_agents, optimize_env_hp=False, optimize_learner_hp=False, evolution=False

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

class MetaLearner(AbstractMetaLearner):

    def __init__():
        pass
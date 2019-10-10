import Learner
from Validation import Validation
from abc import ABC
import torch
import os
from datetime import datetime

# Maybe add a function to dictate the meta learner based on the configs

    # needs to keep track everytime its used so that
    # it always gives a unique id

    # def __init__(self, 
    #             learners : list, 
    #             algorithms : list, 
    #             eval_env : str, 
    #             agents : list, 
    #             elite_agents : list, 
    #             optimize_env_hp : bool, 
    #             optimize_learner_hp : bool, 
    #             evolution : bool,
    #             configs: list):

def initialize_meta(path):

    validate = Validation(path)

    config = validate.learners

    # print(config)

    if config[0]['MetaLearner']['type'] == 'Single':
        return SingleAgentMetaLearner([],
                                    validate.algorithms, # 
                                    config[0]['MetaLearner']['eval_env'],  # the evaluation environment
                                    [],  # this would have to be a list of agent objects
                                    [],  # this would have to be a lsit of elite agents objects
                                    config[0]['MetaLearner']['optimize_env_hp'],
                                    config[0]['MetaLearner']['optimize_learner_hp'],
                                    config[0]['MetaLearner']['evolution'],
                                    config)


class AbstractMetaLearner(ABC):

    def __init__(self, 
                learners : list, 
                algorithms : list, 
                eval_env : str, 
                agents : list, 
                elite_agents : list, 
                optimize_env_hp : bool, 
                optimize_learner_hp : bool, 
                evolution : bool,
                config : list):

        self.learners = learners
        self.algorithms = algorithms
        self.eval_env = eval_env
        self.agents = agents
        self.elite_agents = elite_agents
        self.optimize_env_hp = optimize_env_hp
        self.optimize_learner_hp = optimize_env_hp
        self.evolution = evolution
        self.metaLearnerCount = 0

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



# this is a version of a meta learner that will take a file path to the configuration files
class SingleAgentMetaLearner(AbstractMetaLearner):


    def __init__(self, 
                learners : list, 
                algorithms : list, 
                eval_env : str, 
                agents : list, 
                elite_agents : list, 
                optimize_env_hp : bool, 
                optimize_learner_hp : bool, 
                evolution : bool,
                configs: list):

        super(SingleAgentMetaLearner,self).__init__(learners, 
                                                    algorithms, 
                                                    eval_env, 
                                                    agents, 
                                                    elite_agents, 
                                                    optimize_env_hp, 
                                                    optimize_learner_hp, 
                                                    evolution,
                                                    configs)


        stamp = self.makeDirectory()

        if True:

            #agents, environments, algorithm, data, configs
            self.learner = Learner.Single_Agent_Learner([], [], self.algorithms, [], configs)

            self.learner.launch() 

            self.learner.update()

            self.learner.alg.agents[0].save(3)
                

        else:
            pass
            # end the program somehow

    def makeDirectory(self):

        date, time = str(datetime.now()).split()
        
        date = date[5:]

        time = time[:7]

        stamp = date + '-' + time

        root = 'MetaLearner-' + stamp +'/'
        
        os.system('mkdir ' + root)

        return stamp


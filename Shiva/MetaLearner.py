import Learner
from Validation import Validation
from abc import ABC
import torch
import os
import subprocess
from datetime import datetime

# Maybe add a function to dictate the meta learner based on the configs
# needs to keep track everytime its used so that it always gives a unique id

def initialize_meta(path : "filepath to config file"):

    # validation object reads and checks the configuration files
    validate = Validation(path)

    # get the configurations from the validation object
    config = validate.learners

    # print(config)


    # I could make this a for each loop and be able to generate more than one metalearner
    # makes me think about how the configuration file structure is going to evole

    if config[0]['MetaLearner']['type'] == 'Single':
        return SingleAgentMetaLearner([],
                                    validate.algorithms, # 
                                    config[0]['MetaLearner']['eval_env'],  # the evaluation environment
                                    [],  # this would have to be a list of agent objects
                                    [],  # this would have to be a list of elite agents objects
                                    config[0]['MetaLearner']['optimize_env_hp'],
                                    config[0]['MetaLearner']['optimize_learner_hp'],
                                    config[0]['MetaLearner']['evolution'],
                                    config,
                                    )


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
        self.learnerCount = 0

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
                learners : "list of learner objects but in this case there's only one but there could be more", 
                algorithms : "list of algorithm name strings", 
                eval_env : "the name of the evaluation environment; from config file", 
                agents : "list of agents, in this case there's only one", 
                elite_agents : "list of elite agent objects from evaluation environment", 
                optimize_env_hp : "boolean for whether or not we are optimizing hyperparameters", 
                optimize_learner_hp : "boolean to optimize learner hyperparameters", 
                evolution : "boolean for whether are not we will use evolution",
                configs: "list of all config dictionaries"):

        super(SingleAgentMetaLearner,self).__init__(learners, 
                                                    algorithms, 
                                                    eval_env, 
                                                    agents, 
                                                    elite_agents, 
                                                    optimize_env_hp, 
                                                    optimize_learner_hp, 
                                                    evolution,
                                                    configs)

        # make a directory for this instance of metalearner
        self.root = self.makeDirectory()

        # agents, environments, algorithm, data, configs
        self.learner = Learner.Single_Agent_Learner([], [], self.algorithms, [], configs, self.id_generator(), self.root)

        # initialize the learner
        self.learner.launch() 

        # this method seems misleading or inappropriately named
        # might want to rethink this or live with it
        self.learner.update()

        # save the agent that was trained
        self.learner.alg.agents[0].save(3)


    # This makes the directory timestamped for the instance of Shiva that is running

    # It should put it inside of a runs folder
    def makeDirectory(self):

        # gets the current date and time
        date, time = str(datetime.now()).split()
        
        # we splice out the parts that we want
        date = date[5:] #MM-DD
        time = time[:8]#HH:MM:SS

        # get current path
        path = os.getcwd()

        # make the folder name
        stamp = date + '-' + time
        root = path + '/Shiva/runs/MetaLearner-' + stamp
        
        # make the folder
        subprocess.Popen("mkdir "+root, shell=True)
        
        # return root for reference  
        return root


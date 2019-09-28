from os import listdir
from os.path import isfile, join
import configparser
import ast

# So here I need to loop through 1 or more configuration files or configuration folder
# if a user wants to modularize all the 

# might have to overhaul this whole thing

class Validation():

    def __init__(self, path):
        # store the ini directory for later use
        self.file_path = path
        self.learners = self.read_configs()


    #  this method will assume that we are going to read more than one config file or a folders of config files
    #  if its a single config file then it will be assumed that all the configurations for the learner will be in that file
    #  if its a folder of config files it will expect there to be a config file for each component
    #  I can design this so that read configs will go through each file and identify what kind of learner method needs to be implemented


    def read_configs(self):

        # parser for ini files
        parser = configparser.ConfigParser()

        def section_extracter(section):

            configs = {}

            for option in parser.options(section):
                configs[option] = ast.literal_eval(parser.get(section, option))

            return configs

        learners = []

        # loop through initialization folder
        for f in listdir(self.file_path):

            learner = {}

            # if its a file
            if isfile(join(self.file_path, f)):

                parser.read(join(self.file_path, f))
                
                for section in parser.sections():
                    if section == 'Learner':
                        learner[section] = section_extracter(section)
                    elif section == 'Algorithm':
                        learner[section] = section_extracter(section)
                    elif section == 'Environment':
                        learner[section] = section_extracter(section)
                    elif section == 'Replay_Buffer':
                        learner[section] = section_extracter(section)
                    elif section == 'Agent':
                        learner[section] = section_extracter(section)
                    elif section == 'Network':
                        learner[section] = section_extracter(section)


                # Now I need to figure out how I'm going to handle and extract the configurations

            else:

                for f_ in listdir(join(self.file_path, f)):

                    parser.read(join(self.file_path, f, f_))

                    for section in parser.sections():

                        if section == 'Learner':
                            learner[section] = section_extracter(section)
                        elif section == 'Algorithm':
                            learner[section] = section_extracter(section)
                        elif section == 'Environment':
                            learner[section] = section_extracter(section)
                        elif section == 'Replay_Buffer':
                            learner[section] = section_extracter(section)
                        elif section == 'Agent':
                            learner[section] = section_extracter(section)
                        elif section == 'Network':
                            learner[section] = section_extracter(section)

            learners.append(learner)

        return learners


    # this method will validate the configurations
    # I think there's an order I should do things.
    # we have the check whether or not the hyperparameters are 
    def validate(self):
        

        self.success = False
        
        self.sucess = True



validate = Validation('Initializers')

for l in validate.learners:

    print(l)
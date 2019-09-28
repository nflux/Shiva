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


    #  this method will assume that we are going to read more than one config file or a folders of config files
    #  if its a single config file then it will be assumed that all the configurations for the learner will be in that file
    #  if its a folder of config files it will expect there to be a config file for each component
    #  I can design this so that read configs will go through each file and identify what kind of learner method needs to be implemented


    def read_configs(self, config):

        # parser for ini files
        parser = configparser.ConfigParser()

        # loop through initialization folder
        for f in listdir(self.file_path):

            # if its a file
            if isfile(join(self.file_path, f)):
                parser.read(join(self.file_path, f))



            else:
                pass


        conf_dict = {}
        sections = config.sections()
        for section in sections:
            options = config.options(section)
            for option in options:
                conf_dict[option] = ast.literal_eval(config.get(section, option))

        # I think I want to return a list of configurations
        # It might even be a list of lists of configurations such that everything is modularized
        return conf_dict






    # this method will validate the configurations
    # I think there's an order I should do things.
    # we have the check whether or not the hyperparameters are 
    def validate(self):
        
        
        self.success = False
        
        self.sucess = True

validate = Validation()

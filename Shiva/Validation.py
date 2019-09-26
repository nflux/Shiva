import configparser
import ast

# So here I need to loop through 1 or more configuration files or configuration folder
# if a user wants to modularize all the 

class Validation():

    def __init__(self, config):
        self.conf_dict = self.get_dict(config)

    def get_dict(self, config):
        conf_dict = {}
        sections = config.sections()
        for section in sections:
            options = config.options(section)
            for option in options:
                conf_dict[option] = ast.literal_eval(config.get(section, option))
        return conf_dict

validate = Validation()

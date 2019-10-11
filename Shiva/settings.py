import os, sys
sys.path.append('./modules')

import configparser
import utils.helpers as helpers

START_MODE = 'production'
DIRS_INI = 'dirs.ini'

'''
    Start mode of Shiva
    
    Possible modes:
        productions     create new agents data
        evaluation      uses the agents saved in file with given configs

'''

class Shiva():
    def __init__(self, mode, dirs_ini):
        self.mode = mode
        self.dirs = helpers.config_file_to_dict(dirs_ini)
        self.base_url = os.path.abspath('.')

    def init_path(self):
        return self.base_url + self.dirs['DIRECTORY']['inits']

    def save():
        pass

ctr = Shiva(START_MODE, DIRS_INI)
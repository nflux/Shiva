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

class ShivaFiler():
    def __init__(self, mode, dirs_ini):
        self.mode = mode
        self.dirs = helpers.config_file_to_dict(dirs_ini)
        self._parse_dirs()
    
    def _parse_dirs(self):
        assert ( and 'inits' in self.dirs['DIRECTORY']) or (self.mode == 'evaluation' and 'data' in self.dirs['DIRECTORY']), 'For Shiva mode {} we are missing either "inits" for production or "data" for evaluation '.format(self.mode)
        self.base_url = os.path.abspath('.')
        try:
            self.inits_url = self.base_url + self.dirs['DIRECTORY']['inits']
        except:
            if self.mode == 'production':
                assert False, 'For mode {} we need init files'.format(self.mode)
            else:
                pass
        try:
            self.utils_url = self.base_url + self.dirs['DIRECTORY']['utils']
        except:
            print('No utils were provided')
            pass
        try:
            self.data_url = self.base_url + self.dirs['DIRECTORY']['data']
        except:
            if self.mode == 'evaluation':
                assert False, 'For mode {} we need data files'.format(self.mode)
            else:
                pass
        try:
            self.runs_url = self.base_url + self.dirs['DIRECTORY']['runs']
        except:
            print("No runs provided, setting default at /runs")
            self.runs_url = self.base_url + 'runs'

    def save():
        '''
            Create new directory for the meta learner and all its learners and agents
        '''
        
    def learner():

        

filer = ShivaFiler(START_MODE, DIRS_INI)
import os
import configparser
import inspect
import helpers as helpers

INIT_INI = 'init.ini'

class ShivaAdmin():
    '''
        This object will take care of all the saving/loading of the modules
        Will strictly depend on the configs provided by the INIT_INI file

        Attributes
            @mode
                Possible modes:
                    productions     create new agents per the configs
                    evaluation      uses the agents in file and evaluate them per the configs

            @dirs
                Provided under the INIT_INI[DIRECTORY] section
                Used to provide the absolute directories for the saves/loads

        Public Methods
            @save(caller)
                Ideally we could identify who is the caller and execute the appropiate method
                g.e. identify if the caller is either a MetaLearner or a Learner
            
            @load(caller)
                Same as above
    '''
    def __init__(self, init_dir):
        self._INITS = helpers.config_file_to_dict(init_dir)
        self.mode = self._INITS['SHIVA']['mode']
        self.dirs = self._INITS['DIRECTORY']
        self._parse_dirs()
    
    def _parse_dirs(self):
        assert (self.mode == 'production' and 'inits' in self.dirs) or (self.mode == 'evaluation' and 'data' in self.dirs), "For Shiva mode {} we are missing either 'inits' for production or 'data' for evaluation".format(self.mode)
        self.base_url = os.path.abspath('.')
        try:
            self.inits_url = self.base_url + self.dirs['inits']
        except:
            if self.mode == 'production':
                assert False, "For mode {} we need init files".format(self.mode)
            else:
                pass
        try:
            self.utils_url = self.base_url + self.dirs['utils']
        except:
            print("No utils were provided")
            pass
        try:
            self.data_url = self.base_url + self.dirs['data']
        except:
            if self.mode == 'evaluation':
                assert False, "For mode {} we need data files".format(self.mode)
            else:
                pass
        try:
            self.runs_url = self.base_url + self.dirs['runs']
        except:
            print("No runs provided, setting default at /runs")
            self.runs_url = self.base_url + 'runs'

    def save(self, caller, id=None):
        self.caller = caller
        self.id = id
        if 'metalearner' in inspect.getfile(caller.__class__).lower():
            self._save_meta_learner()
        elif 'learner' in inspect.getfile(caller.__class__).lower():
            self._save_learner()
        else:
            assert False, 'ShivaAdmin saving error'

    def _save_meta_learner(self):
        '''
            Save the config file used
        '''
        print("Saving Meta Learner: ", self.caller)

    def _save_learner(self):
        '''
            Go thru all it's agents and do an agent.save(directory)
        '''
        print("Saving Learner (and it's agents): ", self.caller)

    def load(self, caller, id=None):
        '''
        '''
        pass

    def _load_meta_learner(self, id=None):
        '''
        '''
        pass

    def _load_learner(self, id=None):
        '''
        '''
        pass

shiva = ShivaAdmin(INIT_INI)
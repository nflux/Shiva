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
                Identifies who is the caller and execute the appropiate method
                g.e. identify if the caller is either a MetaLearner or a Learner
            
            @load(caller)
                Same as above, but loads per the config
    '''
    def __init__(self, init_dir):
        self.VALID_MODES = ['production', 'evaluation']
        self._INITS = helpers.load_config_file_2_dict(init_dir)
        assert 'SHIVA' in self._INITS, 'Needs a [SHIVA] section in file {}'.format(init_dir)
        assert 'mode' in self._INITS['SHIVA'] and self._INITS['SHIVA']['mode'] in self.VALID_MODES, 'Need a valid start mode in [SHIVA] section in file {}'.format(init_dir)
        assert 'DIRECTORY' in self._INITS, 'Need [DIRECTORY] section in file {}'.format(init_dir)
        self.mode = self._INITS['SHIVA']['mode']
        self.dirs = self._INITS['DIRECTORY']
        self._parse_dirs()
    
    def _parse_dirs(self):
        '''
            Creates self attributes for the directory folders given in INIT.init
        '''
        assert (self.mode == 'production' and 'inits' in self.dirs) or (self.mode == 'evaluation' and 'data' in self.dirs), "For Shiva mode {} we are missing either 'inits' for production or 'data' for evaluation".format(self.mode)
        self.base_url = os.path.abspath('.')
        for key, folder_name in self.dirs.items():
            directory = self.base_url + folder_name
            setattr(self, key+'_url', directory)
            helpers.make_dir(os.path.join(self.base_url, directory))
            
    def get_inits(self):
        '''
            This method came from the _old/Validation.py

            This method will assume that we are going to read more than one config file or a folders of config files in the INITS folder
            If its a single config file then it will be assumed that all the configurations for the learner will be in that file
            If its a folder of config files it will expect there to be a config file for each component
            I can design this so that read configs will go through each file and identify what kind of learner method needs to be implemented
        '''
        self.meta_learner_config = []
        for f in os.listdir(self.inits_url):
            f = os.path.join(self.inits_url, f)
            if os.path.isfile(f):
                self.meta_learner_config.append(helpers.load_config_file_2_dict(f))
            else:
                for subf in os.listdir(os.path.join(self.inits_url, f)):
                    self.meta_learner_config.append(helpers.load_config_file_2_dict(subf))
        self._curr_meta_learner_dir = helpers.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, 'Metalearner'))
        return self.meta_learner_config

    '''
        Saving Implementations
    '''

    def save(self, caller):
        self.caller = caller
        if 'metalearner' in inspect.getfile(caller.__class__).lower():
            self._save_meta_learner()
        elif 'learner' in inspect.getfile(caller.__class__).lower():
            self._save_learner()
        else:
            assert False, "{} couldn't identify who is trying to save. Only valid for a MetaLearner or Learner classes type".format(self)

    def _save_meta_learner(self):
        print("Saving Meta Learner:", self.caller, '@', self._curr_meta_learner_dir)
        # TODO: save self.meta_laerner_config file
        for learner in self.caller.learners:
            self._save_learner(learner)

    def _save_learner(self, learner=None):
        self._curr_learner_dir = helpers.make_dir_timestamp(os.path.join(self._curr_meta_learner_dir, 'Learner-', self.caller.id if learner is None else learner.id))
        print("Saving Learner:", self.caller if learner is None else learner, '@', self._curr_learner_dir)

    '''
        Loading Implementations
    '''

    def load(self, caller, id=None):
        self.caller = caller
        self.id = id
        if 'metalearner' in inspect.getfile(caller.__class__).lower():
            self._load_meta_learner()
        elif 'learner' in inspect.getfile(caller.__class__).lower():
            self._load_learner()
        else:
            assert False, 'ShivaAdmin saving error'

    def _load_meta_learner(self):
        print("Loading MetaLearner")

    def _load_learner(self):
        print("Loading Learner")

shiva = ShivaAdmin(INIT_INI)
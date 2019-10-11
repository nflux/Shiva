import os
import configparser
import inspect
import helpers as helpers

INIT_INI = 'init.ini'

###########################################################################
# 
#               Shiva Administrative Assistant
#         
###########################################################################

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
        self._curr_meta_learner_dir = ''
        self._curr_learner_dir = {}
        self._curr_agent_dir = {}

        self.VALID_MODES = ['production', 'evaluation']
        self._INITS = helpers.load_config_file_2_dict(init_dir)
        assert 'SHIVA' in self._INITS, 'Needs a [SHIVA] section in file {}'.format(init_dir)
        assert 'mode' in self._INITS['SHIVA'] and self._INITS['SHIVA']['mode'] in self.VALID_MODES, 'Need a valid start mode in [SHIVA] section in file {}'.format(init_dir)
        assert 'DIRECTORY' in self._INITS, 'Need [DIRECTORY] section in file {}'.format(init_dir)
        self.mode = self._INITS['SHIVA']['mode']
        self.need_to_save = self._INITS['SHIVA']['save']
        self.dirs = self._INITS['DIRECTORY']
        self._parse_dirs()

        if self._INITS['SHIVA']['traceback']:
            helpers.warnings.showwarning = helpers.warn_with_traceback

    def __str__(self):
        return "ShivaAdmin"
    
    def _parse_dirs(self):
        '''
            Creates self attributes for the directory folders given in INIT.init
            Creates the folders if needed
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
        if self.need_to_save:
            self._curr_meta_learner_dir = helpers.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, 'ML-'))
        return self.meta_learner_config

    def add_learner_profile(self, learner):
        if not self.need_to_save: pass
        if learner.id not in self._curr_learner_dir:
            self._curr_learner_dir[learner.id] = helpers.make_dir(os.path.join(self._curr_meta_learner_dir, 'L-'+str(learner.id)))
            self._curr_agent_dir[learner.id] = {}
            # print('Learner', learner.id, 'profile added')

    def update_agents_profile(self, learner):
        if not self.need_to_save: pass
        self.add_learner_profile(learner)
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._add_agent_profile(learner.id, agent)
        else:
            self._add_agent_profile(learner.id, learner.agents)

    def _add_agent_profile(self, learner_id: int, agent):
        if not self.need_to_save: pass
        print(self._curr_agent_dir, learner_id)
        if agent.id not in self._curr_agent_dir[learner_id]:
            self._curr_agent_dir[learner_id][agent.id] = helpers.make_dir(os.path.join(self._curr_learner_dir[learner_id], 'Agents', str(agent.id)))
            # print('Agent', agent.id, 'profile added')

    def get_agent_dir(self, learner_id, agent):
        self._add_agent_profile(learner_id, agent)
        return self._curr_agent_dir[learner_id][agent.id]

    '''
        Saving Implementations
    '''

    def save(self, caller):
        if not self.need_to_save: pass
        self.caller = caller
        if 'metalearner' in inspect.getfile(self.caller.__class__).lower():
            self._save_meta_learner()
        elif 'learner' in inspect.getfile(self.caller.__class__).lower():
            self._save_learner()
        else:
            assert False, "{} couldn't identify who is trying to save. Only valid for a MetaLearner or Learner (sub)classes. Got {}".format(self, self.caller)

    def _save_meta_learner(self):
        print("Saving Meta Learner:", self.caller, '@', self._curr_meta_learner_dir)
        # TODO: save self.meta_learner_config file into self._curr_meta_learner_dir
        for learner in self.caller.learners:
            self._save_learner(learner)
    
    def _save_learner(self, learner=None):
        learner = self.caller if learner is None else learner
        self.add_learner_profile(learner)
        # print("Saving Learner:", learner.id, '@', self._curr_learner_dir[learner.id])
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._add_agent_profile(learner.id, agent)
                agent.save(self._curr_agent_dir[learner.id][agent.id], learner.env.get_current_step())
        else:
            self._add_agent_profile(learner.id, learner.agents)
            agent.save(self._curr_agent_dir[learner.id][learner.agents.id], learner.env.get_current_step())

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

###########################################################################
#         
###########################################################################

shiva = ShivaAdmin(INIT_INI)
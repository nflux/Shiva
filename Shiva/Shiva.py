import os
import configparser
import inspect, fnmatch, pickle
import helpers as helpers
from tensorboardX import SummaryWriter

###################################################################################
# 
#   ShivaAdmin
#       Input
#            @INIT_DIR      Absolute address location for initialization file
#         
###################################################################################

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
        self.writer = {}
        
        self.MODE_PRODUCTION = 'production'
        self.MODE_EVALUATION = 'evaluation'
        self.VALID_MODES = [self.MODE_PRODUCTION, self.MODE_EVALUATION]
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
            Set self attributes for the DIRECTORY given in INIT.init
            As well as create the folders if they don't exist
        '''
        assert (self.mode == 'production' and 'inits' in self.dirs) or (self.mode == 'evaluation' and 'data' in self.dirs), "For Shiva mode {} we are missing either 'inits' for production or 'data' for evaluation".format(self.mode)
        self.base_url = os.path.abspath('.')
        for key, folder_name in self.dirs.items():
            directory = self.base_url + folder_name
            setattr(self, key+'_url', directory)
            helpers.make_dir(os.path.join(self.base_url, directory))

    ###
    #    Formats the config file for the MetaLearner
    ###

    def get_inits(self):
        '''
            This method came from the _old/Validation.py

            This method will assume that we are going to read more than one config file or a folders of config files in the INITS folder
            If its a single config file then it will be assumed that all the configurations for the learner will be in that file
            If its a folder of config files it will expect there to be a config file for each component
            I can design this so that read configs will go through each file and identify what kind of learner method needs to be implemented
        '''
        
        if self.mode == self.MODE_PRODUCTION:
            self.meta_learner_config = helpers.parse_configs(self.inits_url)
            return self.meta_learner_config
        elif self.mode == self.MODE_EVALUATION:
            '''
                Check if instances to load are there
                Read them and return
            '''
            self.init_loader = helpers.configparser(self.inits_url)
            meta_learner_to_load = self.init_loader[0]['MetaLearner']['load_run'] + self.output_config_name
            if os.path.isfile(os.path.join(self.runs_url, meta_learner_to_load)):
                print('MetaLearner file requested found at {}'.format(meta_learner_to_load))

    ###
    #   Administration of directory folders
    ###
    
    def add_meta_profile(self, meta_learner, env_name: str=''):
        if self.need_to_save:
            ml_folder_name = 'ML-'+env_name+'-'
            self._curr_meta_learner_dir = helpers.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, ml_folder_name))

    def add_learner_profile(self, learner):
        # print('add_learner_profile', learner)
        if not self.need_to_save: return
        if learner.id not in self._curr_learner_dir:
            self._curr_learner_dir[learner.id] = helpers.make_dir(os.path.join(self._curr_meta_learner_dir, 'L-'+str(learner.id)))
            self._curr_agent_dir[learner.id] = {}
            self.writer[learner.id] = {}
            # print('Learner', learner.id, 'profile added')

    def update_agents_profile(self, learner):
        # print('update_agents_profile', learner)
        if not self.need_to_save: return
        self.add_learner_profile(learner)
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._add_agent_profile(learner, agent)
        else:
            self._add_agent_profile(learner, learner.agents)

    def _add_agent_profile(self, learner, agent):
        if not self.need_to_save: return
        if agent.id not in self._curr_agent_dir[learner.id]:
            self._curr_agent_dir[learner.id][agent.id] = helpers.make_dir(os.path.join(self._curr_learner_dir[learner.id], 'Agents', str(agent.id)))
            self.init_summary_writer(learner, agent)
            # print('Agent', agent.id, 'profile added')

    def get_agent_dir(self, learner, agent):
        # print('get_agent_dir', learner, agent)
        self._add_agent_profile(learner, agent)
        return self._curr_agent_dir[learner.id][agent.id]

    ###
    # Summary Writer Utilities
    ###

    def init_summary_writer(self, learner, agent):
        if not self.need_to_save: return
        self.writer[learner.id][agent.id] =  SummaryWriter(self.get_agent_dir(learner, agent))

    def add_summary_writer(self, learner, agent, scalar_name, value_y, value_x):
        if not self.need_to_save: return
        self.writer[learner.id][agent.id].add_scalar(scalar_name, value_y, value_x)

    ###
    # Saving Implementations
    ###

    def save(self, caller):
        if not self.need_to_save: return
        self.caller = caller
        if 'metalearner' in inspect.getfile(self.caller.__class__).lower():
            self._save_meta_learner()
        elif 'learner' in inspect.getfile(self.caller.__class__).lower():
            self._save_learner()
        else:
            assert False, "{} couldn't identify who is trying to save. Only valid for a MetaLearner or Learner (sub)classes. Got {}".format(self, self.caller)

    def _save_meta_learner(self):
        print("Saving Meta Learner:", self.caller, '@', self._curr_meta_learner_dir)
        # create the meta learner configs folder
        helpers.make_dir(os.path.join(self._curr_meta_learner_dir, 'configs'))
        # save each config file
        for cf in self.caller.configs:
            _filename_ = os.path.split(cf['_filename_'])[-1]
            helpers.save_dict_2_config_file(cf, os.path.join(self._curr_meta_learner_dir, 'configs', _filename_))
        # save each learner
        for learner in self.caller.learners:
            self._save_learner(learner)

    def _save_learner(self, learner=None):
        learner = self.caller if learner is None else learner
        self.add_learner_profile(learner)
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._add_agent_profile(learner, agent)
                agent_url = self._curr_agent_dir[learner.id][agent.id]
                agent.save(agent_url, learner.env.get_current_step())
                with open(agent_url+'/cls.pickle', 'wb') as handle:
                    pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self._add_agent_profile(learner, learner.agents)
            agent_url = self._curr_agent_dir[learner.id][agent.agents.id]
            agent.save(agent_url, learner.env.get_current_step())
            with open(agent_url+'/cls.pickle', 'wb') as handle:
                pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###
    # Loading Implementations
    ###

    def load(self, caller, id=None):
        self.caller = caller
        self.id = id
        if 'metalearner' in inspect.getfile(caller.__class__).lower():
            self._load_meta_learner()
        elif 'learner' in inspect.getfile(caller.__class__).lower():
            self._load_learner()
        else:
            assert False, '{} loading error'.format(self)

    def _load_meta_learner(self):
        print("Loading MetaLearner")

    def _load_learner(self):
        print("Loading Learner")

    def _load_agents(self, path):
        agents = []
        agents_pickles = helpers.find_pattern_in_path(path, '*.pickle')
        agents_policies = helpers.find_pattern_in_path(path, '*.pth')
        assert len(agents_pickles) > 0, "No agents found in {}".format(path)
        for agent_pickle, agent_policy in zip(agents_pickles, agents_policies):
            with open(agent_pickle, 'rb') as handle:
                _new_agent = pickle.load(handle)
                _new_agent.load(agent_policy)
                agents.append(_new_agent)
        return agents

###########################################################################
#         
###########################################################################
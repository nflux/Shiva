import os
import configparser
import inspect
import helpers as helpers
from tensorboardX import SummaryWriter

###################################################################################
# 
#   ShivaAdmin
#       This administrator class is a filing helper for the Shiva framework
#
#       Part of it's tasks are:
#       
#           - Track and create directories for the MetaLearners, Learners, Agents
#           - Perform the mechanics of saving and loading of MetaLearners, Learners, Agents which include
#                   - The config files used
#                   - The Agent class pickle
#                   - The Agent networks (implemented at the Agent level due to the different Agents specifications)
#           - Add a SummaryWriter for the Agents and the metrics for a given agent
#
#       The folders structure being used are given by the input @init_dir on ShivaAdmin instantiation
#       Requirements of the .INI file
#
#            [SHIVA]
#            save =              True           Saving option
#            traceback =         True           Debugging traceback option
#
#            [DIRECTORY]
#            runs =              "/runs"        Folder where the runs will be saved
#            inits =             "/inits"       Config files that will be passed to the MetaLearner
#            modules =           "/modules"     Folder where the MetaLearner, Learner, Agents, Environment, etc..
#            utils =             "/utils"       Utilities/helpers folder
#            data =              "/data"        Some other data?
#
#         
###################################################################################

class ShivaAdmin():
    def __init__(self, init_dir):
        '''
            Input
                @init_dir       Absolute path for the ShivaAdmin ini file declared in settings.py
        '''
        self._curr_meta_learner_dir = ''
        self._curr_learner_dir = {}
        self._curr_agent_dir = {}
        self.writer = {}
        
        self._INITS = helpers.load_config_file_2_dict(init_dir)
        
        assert 'SHIVA' in self._INITS, 'Needs a [SHIVA] section in file {}'.format(init_dir)
        assert 'DIRECTORY' in self._INITS, 'Need [DIRECTORY] section in file {}'.format(init_dir)
        
        self.need_to_save = self._INITS['SHIVA']['save']
        self.dirs = self._INITS['DIRECTORY']
        self._set_dirs_attrs()

        if self._INITS['SHIVA']['traceback']:
            helpers.warnings.showwarning = helpers.warn_with_traceback

    def __str__(self):
        return "ShivaAdmin"
    
    def _set_dirs_attrs(self) -> None:
        '''
            Set self attributes for accessing directories
            These directories come from the section DIRECTORY in the init.ini (or other given by user at settings.py
            The procedure as well creates the needed folders for this directories if they don't exist
        '''
        self.base_url = os.path.abspath('.')
        for key, folder_name in self.dirs.items():
            directory = self.base_url + folder_name
            setattr(self, key+'_url', directory)
            helpers.make_dir(os.path.join(self.base_url, directory))

    def get_inits(self) -> list:
        '''
            This procedure reads the *.ini files in path given by user at the init.ini file, section [DIRECTORY], attribute INITS

            Returns
                A list of config dictionaries
        '''
        self.meta_learner_config = helpers.parse_configs(self.inits_url)
        return self.meta_learner_config
    
    def add_meta_profile(self, meta_learner, env_name: str='') -> None:
        '''
            This method would be called by a Meta Learner in order to add himself
            Is needed in order to keep track of the Meta Learner directory.
            Creates a folder for it if save flag is True.

            Input
                @meta_learner       Meta Learner instance to be filed
                @env_name           Environment name to append to the Meta Learner folder
        '''
        if self.need_to_save:
            ml_folder_name = 'ML-'+env_name+'-'
            self._curr_meta_learner_dir = helpers.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, ml_folder_name))

    def add_learner_profile(self, learner) -> None:
        '''
            Needed to keep track of the Learner and it's directory.
            Creates a folder for it if save flag is True.

            Input
                @learner            Learner instance to be filed
        '''
        if not self.need_to_save: return
        if learner.id not in self._curr_learner_dir:
            self._curr_learner_dir[learner.id] = helpers.make_dir(os.path.join(self._curr_meta_learner_dir, 'L-'+str(learner.id)))
            self._curr_agent_dir[learner.id] = {}
            self.writer[learner.id] = {}

    def update_agents_profile(self, learner):
        '''
            This procedure adds all the agents profile inside the learner

            Input
                @learner            Learner instance
        '''
        if not self.need_to_save: return
        self.add_learner_profile(learner) # will only add if was not profiled before
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._add_agent_profile(learner, agent)
        else:
            self._add_agent_profile(learner, learner.agents)

    def _add_agent_profile(self, learner, agent):
        '''
            Needed to keep track of the agents and it's directory.
            Creates a folder for it if save flag is True.

            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent instance to be filed
        '''
        if not self.need_to_save: return
        if agent.id not in self._curr_agent_dir[learner.id]:
            self._curr_agent_dir[learner.id][agent.id] = helpers.make_dir(os.path.join(self._curr_learner_dir[learner.id], 'Agents', str(agent.id)))
            self.init_summary_writer(learner, agent)

    def get_agent_dir(self, learner, agent) -> str:
        '''
            Returns the absolute address location of the given agent
            
            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent instance
        '''
        self._add_agent_profile(learner, agent)
        return self._curr_agent_dir[learner.id][agent.id]

    def init_summary_writer(self, learner, agent) -> None:
        '''
            Instantiates the SummaryWriter for the given agent

            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent who we want to records the metrics
        '''
        if not self.need_to_save: return
        self.writer[learner.id][agent.id] =  SummaryWriter(self.get_agent_dir(learner, agent))

    def add_summary_writer(self, learner, agent, scalar_name, value_y, value_x) -> None:
        '''
            Adds a metric to the tensorboard of the given agent

            Input
                @learner            Learner instance owner of the agent
                @agent              Agent who we want to add
                @scalar_name        Metric name
                @value_y            Usually the metric
                @value_x            Usually time
        '''
        if not self.need_to_save: return
        self.writer[learner.id][agent.id].add_scalar(scalar_name, value_y, value_x)

    def save(self, caller) -> None:
        '''
            This procedure is for the MetaLearner (or Learner) to save all it's configurations and agents
            If a MetaLearner is the caller, the saving will cascade along all the Learner that it has, and all the agents inside the Learner
            
            Requirement
                The caller, before saving, must have added his profile, if not, an error will be thrown

            Input
                @caller             Either a MetaLearner or Learner instance
        '''
        if not self.need_to_save: return
        self.caller = caller
        if 'metalearner' in inspect.getfile(self.caller.__class__).lower():
            self._save_meta_learner()
        elif 'learner' in inspect.getfile(self.caller.__class__).lower():
            self._save_learner()
        else:
            assert False, "{} couldn't identify who is trying to save. Only valid for a MetaLearner or Learner subclasses. Got {}".format(self, self.caller)

    def _save_meta_learner(self) -> None:
        '''
            Mechanics of saving a MetaLearner
                1-  Saves each of the config files used
                2-  Then iterates over all it's Learners to save them

            No input because we are only handling only ONE MetaLearner
        '''
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

    def _save_learner(self, learner=None) -> None:
        '''
            Mechanics of saving a Learner
                1-  Pickles the Learner class and save attributes
                2-  Save all the agents inside

            Input
                @learner        Learner instance we want to save
        '''
        learner = self.caller if learner is None else learner
        self.add_learner_profile(learner) # will only add if was not profiled before
        # save learner pickle and attributes
        learner_path = self._curr_learner_dir[learner.id]
        helpers.save_pickle_obj(learner, os.path.join(learner_path, 'learner_cls.pickle'))
        # save agents
        if type(learner.agents) == list:
            for agent in learner.agents:
                self._save_agent(learner, agent)
        else:
            self._save_agent(learner, learner.agents)

    def _save_agent(self, learner, agent):
        '''
            Mechanics of saving a Agent
                1-  Pickles the agent object and save attributes
                2-  Uses the save() method from the Agent class because Agents could have diff network structures

            Input
                @learner        Learner who owns the agent
                @agent          Agent we want to save
        '''
        self._add_agent_profile(learner, agent)
        agent_path = self._curr_agent_dir[learner.id][agent.id]
        helpers.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))
        agent.save(agent_path, learner.env.get_current_step())

    def load(self, caller, id=None):
        '''
            TODO
                I don't see a point of having this function for now
                I guess will have the specific implementations as below
                To think about..
        '''
        self.caller = caller
        self.id = id
        if 'metalearner' in inspect.getfile(caller.__class__).lower():
            self._load_meta_learner()
        elif 'learner' in inspect.getfile(caller.__class__).lower():
            self._load_learner()
        else:
            assert False, '{} loading error'.format(self)

    def _load_meta_learner(self):
        '''
            TODO
                Implement once needed in order to see what's the best approach
                
            Returns
                MetaLearner instance
        '''
        print("Loading MetaLearner")

    def _load_learner(self, path: str) -> object:
        '''
            The procedure will load only one single Learner if found in the given path

            Input
                @path       Path where the learner files will be located

            Returns
                Learner
        '''
        learner_pickle = helpers.find_pattern_in_path(path, 'learner_cls.pickle')
        assert len(learner_pickle) > 0, "No learners found in {}".format(path)
        assert len(learner_pickle) == 1, "{} learner classes were found in {}".format(str(len(learner_pickle)), path)
        learner_pickle = learner_pickle[0]
        print('Loading Learner\n\t{}\n'.format(learner_pickle))
        learner_path, _ = os.path.split(learner_pickle)
        _new_learner = helpers.load_pickle_obj(learner_pickle)
        _new_learner.agents = self._load_agents(learner_path)
        return _new_learner

    def _load_agents(self, path) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load all those agents

            Input
                @path       Path where the agents files will be located
        '''
        agents = []
        agents_pickles = helpers.find_pattern_in_path(path, 'agent_cls.pickle')
        agents_policies = helpers.find_pattern_in_path(path, 'policy.pth')
        assert len(agents_pickles) > 0, "No agents found in {}".format(path)
        print("Loading Agents..")
        for agent_pickle, agent_policy in zip(agents_pickles, agents_policies):
            print("\t{}\n\t{}\n\n".format(agent_pickle, agent_policy))
            _new_agent = helpers.load_pickle_obj(agent_pickle)
            _new_agent.load_net(agent_policy)
            agents.append(_new_agent)
        return agents

###########################################################################
#         
###########################################################################
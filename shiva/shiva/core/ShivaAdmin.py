import os
import configparser
import inspect
import shiva.helpers.dir_handler as dh
import shiva.helpers.file_handler as fh
import shiva.helpers.config_handler as ch
import shiva.helpers.misc as misc
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
    def __init__(self):
        pass

    def init(self, config):
        '''
            Input
                @config       Dictionary of the Shiva Admin config
        '''
        self.config = config
        
        self.need_to_save = self.config['save']
        self.traceback = self.config['traceback']
        self.dirs = self.config['directory']
        self._set_dirs_attrs()

        if self.traceback:
            misc.warnings.showwarning = misc.warn_with_traceback

        self._curr_meta_learner_dir = ''
        self._curr_learner_dir = {}
        self._curr_agent_dir = {}
        self.writer = {}

    def __str__(self):
        return "ShivaAdmin"
    
    def _set_dirs_attrs(self) -> None:
        '''
            Set self attributes for accessing directories
            These directories come from the section DIRECTORY in the init.ini (or other given by user at settings.py
            The procedure as well creates the needed folders for this directories if they don't exist
        '''
        assert 'runs' in self.dirs, "Need the 'runs' key on the 'directory' attribute on the [admin] section of the config"
        self.base_url = os.path.abspath('.')
        for key, folder_name in self.dirs.items():
            directory = self.base_url + folder_name
            setattr(self, key+'_url', directory)
            dh.make_dir(os.path.join(self.base_url, directory), overwrite=True)

    # def get_inits(self) -> list:
    #     '''
    #         This procedure reads the *.ini files in path given by user at the init.ini file, section [DIRECTORY], attribute INITS
    #
    #         Returns
    #             A list of config dictionaries
    #     '''
    #     self.meta_learner_config = ch.parse_configs(self.inits_url)
    #     return self.meta_learner_config
    
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
            ml_folder_name = env_name
            self._curr_meta_learner_dir = dh.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, ml_folder_name))
            print("New MetaLearner @ {}".format(self._curr_meta_learner_dir))

    def add_learner_profile(self, learner) -> None:
        '''
            Needed to keep track of the Learner and it's directory.
            Creates a folder for it if save flag is True.

            Input
                @learner            Learner instance to be filed
        '''
        if not self.need_to_save: return
        if learner.id not in self._curr_learner_dir:
            self._curr_learner_dir[learner.id] = dh.make_dir(os.path.join(self._curr_meta_learner_dir, 'L-'+str(learner.id)))
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
        self._save_learner_agents_mechanics(learner)

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
            self._curr_agent_dir[learner.id][agent.id] = dh.make_dir(os.path.join(self._curr_learner_dir[learner.id], 'Agents', str(agent.id)))
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
        # print('before writing learner_id {}\tagent_id {}\tscalar {}\tvalue_y {}\tvalue_x {}'.format(learner.id, agent.id, scalar_name, value_y, value_x))
        self.writer[learner.id][agent.id].add_scalar(scalar_name, value_y, value_x)
        # print('after writing!')

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
        print("Saving Meta Learner @ {}".format(self._curr_meta_learner_dir))
        # create the meta learner configs folder
        dh.make_dir(os.path.join(self._curr_meta_learner_dir, 'configs'))
        # save each config file
        if type(self.caller.config) == dict:
            cf = self.caller.config
            _filename_ = os.path.split(cf['_filename_'])[-1]
            ch.save_dict_2_config_file(cf, os.path.join(self._curr_meta_learner_dir, 'configs', _filename_))
        elif type(self.caller.config) == list:
            for cf in self.caller.config:
                _filename_ = os.path.split(cf['_filename_'])[-1]
                ch.save_dict_2_config_file(cf, os.path.join(self._curr_meta_learner_dir, 'configs', _filename_))
        else:
            assert False, "MetaLearner.config must be a list or a dictionary"
        # save each learner
        try:
            for learner in self.caller.learners:
                self._save_learner(learner)
        except AttributeError:
            self._save_learner(self.caller.learner)

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
        fh.save_pickle_obj(learner, os.path.join(learner_path, 'learner_cls.pickle'))
        # save agents
        self._save_learner_agents_mechanics(learner)

    def _save_learner_agents_mechanics(self, learner):
        '''
            This procedure is it's own function because is used in other parts of the code
            If attribute learner.agents is a valid attribute, saves them (if iterable) or assumes is 1 agent
            If attribute learner.agents is not valid, will try with learner.agent

            Input
                @learner            Learner instance who contains the agents
        '''
        try:
            if type(learner.agents) == list:
                for agent in learner.agents:
                    self._save_agent(learner, agent)
                    self._save_buffer(learner,agent)
            else:
                self._save_agent(learner, learner.agents)
                self._save_buffer(learner, learner.agents)
        except AttributeError: # when learner has only 1 agent
            self._save_agent(learner, learner.agent)
            self._save_buffer(learner, learner.agent)

    def _save_agent(self, learner, agent):
        '''
            Mechanics of saving a Agent
                1-  Pickles the agent object and save attributes
                2-  Uses the save() method from the Agent class because Agents could have diff network structures

            Input
                @learner        Learner who owns the agent
                @agent          Agent we want to save
        '''
        if not self.need_to_save: return
        self._add_agent_profile(learner, agent)
        agent_path = os.path.join(self._curr_agent_dir[learner.id][agent.id], 'Ep'+str(learner.env.done_count))
        agent_path = dh.make_dir(agent_path)
        fh.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))
        agent.save(agent_path, learner.env.get_current_step())

    def _save_buffer(self,learner, agent):
        ''' 
            Mechanics of saving a Replay Buffer

            1 - pickles the buffer object and saves it in the corresponding agent's folder
        '''
        buffer_path = self._curr_agent_dir[learner.id][agent.id]
        fh.save_pickle_obj(learner.buffer, os.path.join(buffer_path, 'buffer.pickle'))

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
        learner_pickle = dh.find_pattern_in_path(path, 'learner_cls.pickle')
        assert len(learner_pickle) > 0, "No learners found in {}".format(path)
        assert len(learner_pickle) == 1, "{} learner classes were found in {}".format(str(len(learner_pickle)), path)
        learner_pickle = learner_pickle[0]
        print('Loading Learner\n\t{}\n'.format(learner_pickle))
        learner_path, _ = os.path.split(learner_pickle)
        _new_learner = fh.load_pickle_obj(learner_pickle)
        _new_learner.agents = self._load_agents(learner_path)
        _new_learner.buffer = self._load_buffer(learner_path)
        return _new_learner

    def _load_agents(self, path) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load all those agents

            Input
                @path       Path where the agents files will be located
        '''
        agents = []
        agents_pickles = dh.find_pattern_in_path(path, 'agent_cls.pickle')
        agents_policies = dh.find_pattern_in_path(path, '.pth')
        assert len(agents_pickles) > 0, "No agents found in {}".format(path)
        print("Loading multiple agents is not yet so friendly. Try using shiva._load_agent()")
        print("Loading Agents..")
        for agent_pickle, agent_policy in zip(agents_pickles, agents_policies):
            print("\t{}\n\t{}\n".format(agent_pickle, agent_policy))
            _new_agent = fh.load_pickle_obj(agent_pickle)
            _new_agent.load_net(agent_policy)
            agents.append(_new_agent)
        return agents


    def _load_agent(self, path) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load a single agent

            InputW
                @path       Path where the agents files will be located
        '''
        agent_pickle = dh.find_pattern_in_path(path, 'agent_cls.pickle')
        agent_policy = dh.find_pattern_in_path(path, '.pth')
        assert len(agent_pickle) > 0, "No agent found in {}".format(path)
        assert len(agent_pickle) == 1, "Multiple agent_cls.pickles found. Try using shiva._load_agents()"
        print("Loading Agent..")
        print("\t{}\n\twith {} networks\n".format(agent_pickle, len(agent_policy)))
        _new_agent = fh.load_pickle_obj(agent_pickle[0])
        for policy_file in agent_policy:
            _new_agent.load_net(policy_file)
        return _new_agent


    def _load_buffer(self, path) -> list:
        '''
            For now, we have only 1 buffer per learner
            
            Input
                @path       Learner path
        '''

        buffer_pickle = dh.find_pattern_in_path(path, 'buffer.pickle')[0]
        assert len(buffer_pickle) > 0, "No buffer found in {}".format(path)
        print("Loading Buffer..")
        print("\t{}\n".format(buffer_pickle))
        return fh.load_pickle_obj(buffer_pickle)

    

###########################################################################
#         
###########################################################################
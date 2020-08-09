import os, re
import configparser
import inspect
import numpy as np
from tensorboardX import SummaryWriter
import torch

import shiva.helpers.dir_handler as dh
import shiva.helpers.file_handler as fh
import shiva.helpers.config_handler as ch
import shiva.helpers.misc as misc

###################################################################################
#
#   ShivaAdmin
#       This administrator class is a filing helper for the Shiva framework
#
#       Part of it's current tasks are:
#           - Track and create directories for the MetaLearners, Learners, Agents
#           - Perform the mechanics of saving and loading of MetaLearners, Learners, Agents which include
#                   - The config file used for the Learner session
#                   - The class pickles (Learner, Agent, Buffer)
#                   - The Agent networks (implemented at the Agent level due to the different Agents specifications, like having actor, critic, target, etc..)
#           - Add a SummaryWriter for the Agents and the metrics for a given agent
#
#       Config Requirements:
#
#            [SHIVA]
#            save =              True                   Saving option
#            traceback =         True                   Debugging traceback option
#            directory =         { 'runs': '/runs' }    Dictionary of folders where data will be stored
#
###################################################################################

class ShivaAdmin():

    __folder_name__ = {
        'config':       'configs',
        'metalearner':  '{algorithm}-{env}',
        'learner':      'L{id}',
        'summary':      'Tensorboards',
        'checkpoint':   'Ep{ep_num}',
        'learner_data': 'Learner_Data',
        'agent':        '{id}-{role}',
        'latest':      'last'
    }

    def __init__(self, logger, config=None):
        self.logger = logger
        if config is not None:
            self.init(config)
        self.base_url = os.path.abspath('.')

    def init(self, configs):
        '''
            Input
                @config       Dictionary of the Shiva Admin config
        '''
        self.configs = configs
        self.logger.configs = configs
        if 'Admin' in configs:
            {setattr(self, k, v) for k, v in configs['Admin'].items()}
            self.need_to_save = self.configs['Admin']['save']
            self.traceback = self.configs['Admin']['traceback']
            self.dirs = self.configs['Admin']['directory']
        else:
            {setattr(self, k, v) for k, v in configs.items()}
            self.need_to_save = self.configs['save']
            self.traceback = self.configs['traceback']
            self.dirs = self.configs['directory']

        self._set_dirs_attrs()

        if self.traceback:
            misc.warnings.showwarning = misc.warn_with_traceback

        self._meta_learner_dir = ''
        self._learner_dir = {}
        self._agent_dir = {}
        self.writer = {}

    def _set_dirs_attrs(self) -> None:
        '''
            Set self attributes for accessing directories
            These directories come from the section DIRECTORY in the init.ini (or other given by user at settings.py
            The procedure as well creates the needed folders for this directories if they don't exist
        '''
        assert 'runs' in self.dirs, "Need the 'runs' key on the 'directory' attribute on the [admin] section of the config"
        for key, folder_name in self.dirs.items():
            directory = self.base_url + folder_name
            setattr(self, key+'_url', directory)
            if not self.need_to_save: continue
            dh.make_dir(os.path.join(self.base_url, directory), use_existing=True)

    def add_meta_profile(self, meta_learner, folder_name: str=None, use_existing=False) -> None:
        '''
            This method would be called by a Meta Learner in order to add himself
            Is needed in order to keep track of the Meta Learner directory.
            Creates a folder for it if save flag is True.
            Input
                @meta_learner       Meta Learner instance to be filed
                @folder_name              String to use as the folder name
        '''
        if not self.need_to_save: return
        if folder_name is None:
            folder_name = self.__folder_name__['metalearner'].format(algorithm=meta_learner.config['Algorithm']['type'], env=meta_learner.config['Environment']['env_name'])
        new_dir = dh.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, folder_name), use_existing=use_existing)
        self._meta_learner_dir = new_dir
        self.log("New MetaLearner @ {}".format(self._meta_learner_dir), to_print=True, verbose_level=1)

    def get_meta_url(self):
        return self._meta_learner_dir

    def add_learner_profile(self, learner, function_only=False) -> None:
        '''
            Needed to keep track of the Learner and it's directory.
            Creates a folder for it if save flag is True.
            Input
                @learner            Learner instance to be saved
                @function_only      When we only want to use this functionality without profiling, used for distributed processes mode
        '''
        if not self.need_to_save: return
        if learner.id not in self._learner_dir:
            if function_only:
                # need to create a new fake "meta profile" to create root session folder structures for this learner
                if 'exec' in learner.configs['Environment']:
                    env_name = learner.configs['Environment']['exec'].split('/')[-1].replace('.app', '').replace('.86_64', '')
                else:
                    env_name = learner.configs['Environment']['env_name']
                folder_name = self.__folder_name__['metalearner'].format(algorithm=self.configs['Algorithm']['type'], env=env_name)
                self.add_meta_profile(None, folder_name, use_existing=True)
            self._learner_dir[learner.id] = {}
            new_dir = dh.make_dir( os.path.join(self._meta_learner_dir, self.__folder_name__['learner'].format(id=str(learner.id))) )
            self._learner_dir[learner.id]['base'] = new_dir
            self._learner_dir[learner.id]['checkpoint'] = [] # keep track of each checkpoint directory
            latest_dir = dh.make_dir( os.path.join(self._learner_dir[learner.id]['base'], self.__folder_name__['latest']), use_existing=True )
            self._learner_dir[learner.id]['latest'] = latest_dir
            new_dir = dh.make_dir( os.path.join(self._learner_dir[learner.id]['base'], self.__folder_name__['summary']) )
            self._learner_dir[learner.id]['summary'] = new_dir
            self._agent_dir[learner.id] = {}
            self.writer[learner.id] = {} # this will be a dictionary whos key is an agent.id that maps to a unique tensorboard file for that agent

    def checkpoint(self, learner, checkpoint_num=None, function_only=False, use_temp_folder=False):
        '''
            This procedure:
                1. Adds Learner profile if not existing
                2. Creates a directory for the new checkpoint
                3. Saves current state of agents, learner and config in checkpoint folder
            Input
                @learner            Learner instance
                @checkpoint_num     Optional checkpoint number given, used for distributed processes scenario and running locally
                @function_only      When we only want to use this functionality, used for distributed processes and running locally - future to be used on the cloud?
        '''
        if not self.need_to_save: return
        self.add_learner_profile(learner, function_only) # will only add if was not profiled before
        if checkpoint_num is None:
            checkpoint_num = learner.env.done_count
        # create checkpoint folder
        self.use_temp_folder = use_temp_folder
        if self.use_temp_folder:
            checkpoint_dir = self._learner_dir[learner.id]['latest']
        else:
            # use_existing=True so that in case the Learner did not received a new trajectory
            # but still attemps to save, we must be using the same folder
            checkpoint_dir = dh.make_dir(os.path.join( self._learner_dir[learner.id]['base'], self.__folder_name__['checkpoint'].format(ep_num=str(checkpoint_num)) ), use_existing=True)
        self._learner_dir[learner.id]['checkpoint'].append(checkpoint_dir)
        self._save_learner(learner, checkpoint_num)
        self._save_learner_agents(learner, checkpoint_num)

    def get_last_checkpoint(self, learner):
        if len(self._learner_dir[learner.id]['checkpoint']) == 0:
            # no checkpoint available
            return False
        else:
            return self._learner_dir[learner.id]['checkpoint'][-1]

    def get_learner_url(self, learner):
        return self._learner_dir[learner.id]['base']

    def get_latest_directory(self, learner):
        assert learner.id in self._learner_dir, "Learner was not profiled by ShivaAdmin, try calling Admin.add_learner_profile at initialization "
        return self._learner_dir[learner.id]['latest']

    def get_learner_url_summary(self, learner: object=None, some_path: str=None):
        if learner is not None:
            return self._learner_dir[learner.id]['summary']
        else:
            # extract the base url from the given sample @some_path
            # print("Got {}".format(some_path))

            # match = re.search(rx, some_path)
            try:
                rx = "\w+-\w+-\d{2}-\d{2}-\d{4}\/"
                _literal_found = re.findall(rx, some_path)[-1] # get the last directory with this format
            except IndexError:
                rx = "\d{2}-\d{2}-\d{4}\/"
                _literal_found = re.findall(rx, some_path)[-1]  # get the last directory with this format
            match = re.search(_literal_found, some_path)
            _start, _end = match.span()[0], match.span()[1]
            _return = some_path.replace(some_path[_end:], '')
            # print("Given dir: {}".format(_return))
            return _return

    def _add_agent_checkpoint(self, learner, agent):
        '''
            Creates the corresponding folder for the agent checkpoint
            Instantiates the Tensorboard SummaryWriter if doesn't exists
            Input
                @learner            Learner instance ref owner of the Agent
                @agent              Agent instance ref to be saved
        '''
        if not self.need_to_save: return
        new_dir = dh.make_dir( os.path.join( self._learner_dir[learner.id]['checkpoint'][-1], self.__folder_name__['agent'].format(id=str(agent.id), role=agent.role) ), use_existing=True )
        if agent.id not in self._agent_dir[learner.id]:
            self._agent_dir[learner.id][agent.id] = []
        self._agent_dir[learner.id][agent.id].append(new_dir)
        self.init_summary_writer(learner, agent) # just in case it's a new agent?

    def get_new_agent_dir(self, learner, agent) -> str:
        '''
            Creates a new checkpoint directory for the agent and returns it
            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent instance
        '''
        self._add_agent_checkpoint(learner, agent)
        if self.use_temp_folder:
            return os.path.join(self._learner_dir[learner.id]['latest'], self.__folder_name__['agent'].format(id=str(agent.id), role=agent.role) )
        else:
            return self._agent_dir[learner.id][agent.id][-1]

    def init_summary_writer(self, learner, agent) -> None:
        '''
            Instantiates the SummaryWriter for the given agent
            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent who we want to records the metrics
        '''
        if not self.need_to_save: return
        if agent.id not in self.writer[learner.id]:
            new_dir = dh.make_dir( os.path.join( self._learner_dir[learner.id]['summary'], self.__folder_name__['agent'].format(id=str(agent.id), role=agent.role) ) )
            self.writer[learner.id][agent.id] = SummaryWriter(
                logdir = new_dir,
                # filename_suffix = '-' + self.__folder_name__['agent'].format(id=str(agent.id))
            )

    def add_summary_writer(self, learner, agent, scalar_name, value_y, value_x) -> None:
        '''
            Adds a metric to the tensorboard of the given agent
            Input
                @learner            Learner instance owner of the agent
                @agent              Agent who we want to add, or agent_id
                @scalar_name        Metric name
                @value_y            Usually the metric
                @value_x            Usually time
        '''
        if not self.need_to_save: return
        # self.log("{} {} {} {} {}".format(learner.id, agent, scalar_name, value_y, value_x), verbose_level=1)
        if type(agent) == np.int64 or type(agent) == int:
            '''Agent ID was sent'''
            self.writer[learner.id][agent].add_scalar(scalar_name, value_y, value_x)
        else:
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
            # self._save_meta_learner()
            pass
        elif 'learner' in inspect.getfile(self.caller.__class__).lower():
            # self._save_learner()
            self.checkpoint(self.caller)
        else:
            assert False, "{} couldn't identify who is trying to save. Only valid for a MetaLearner or Learner subclasses, but got {}".format(self, self.caller)

    def _save_meta_learner(self) -> None:
        '''
            Mechanics of saving a MetaLearner
            Not sure what do we want to save here
        '''
        assert False, "Not implemented"
        # self.log("Saving Meta Learner @ {}".format(self._meta_learner_dir))
        # # create the meta learner configs folder
        # dh.make_dir(os.path.join(self._meta_learner_dir, self.__folder_name__['config']))

        # save each config file
        # if type(self.caller.config) == dict:
        #     cf = self.caller.config
        #     filename = os.path.split(cf['_filename_'])[-1]
        #     ch.save_dict_2_config_file(cf, os.path.join(self._meta_learner_dir, self.__folder_name__['config'], filename))
        # elif type(self.caller.config) == list:
        #     for cf in self.caller.config:
        #         filename = os.path.split(cf['_filename_'])[-1]
        #         ch.save_dict_2_config_file(cf, os.path.join(self._meta_learner_dir, self.__folder_name__['config'], filename))
        # else:
        #     assert False, "MetaLearner.config must be a list or a dictionary"
        # save each learner

        # try:
        #     for learner in self.caller.learners:
        #         self._save_learner(learner)
        # except AttributeError:
        #     self._save_learner(self.caller.learner)

    def _save_learner(self, learner=None, checkpoint_num=None) -> None:
        '''
            Mechanics of saving a Learner
                1.  Pickles the Learner class
                2.  Pickles the Buffer
                3.  Saves the Learners config
            Input
                @learner        Learner instance we want to save
        '''
        if self.use_temp_folder:
            # no need to save Learner data when using temp folder
            return
        learner = self.caller if learner is None else learner
        self.add_learner_profile(learner) # will only add if was not profiled before
        # save learner pickle
        learner_data_dir = dh.make_dir( os.path.join(self._learner_dir[learner.id]['checkpoint'][-1], self.__folder_name__['learner_data']), use_existing=True )
        # fh.save_pickle_obj(learner, os.path.join(learner_data_dir, 'learner_cls.pickle'))
        # save buffer
        # fh.save_pickle_obj(learner.buffer, os.path.join(learner_data_dir, 'buffer_cls.pickle'))
        # save learner current config status
        if type(learner.configs) == dict:
            cf = learner.configs
            filename = os.path.split(cf['_filename_'])[-1]
            ch.save_dict_2_config_file(cf, os.path.join(learner_data_dir, filename))
        elif type(learner.configs) == list:
            for cf in learner.config:
                filename = os.path.split(cf['_filename_'])[-1]
                ch.save_dict_2_config_file(cf, os.path.join(learner_data_dir, filename))

    def _save_learner_agents(self, learner, checkpoint_num=None):
        '''
            This procedure is it's own function because is used in other parts of the code
            If attribute learner.agents is a valid attribute, saves them (if iterable) or assumes is 1 agent
            If attribute learner.agents is not valid, will try with learner.agent
            Input
                @learner            Learner instance who contains the agents
                @checkpoint_num     Checkpoint numbered used, if not given, will try to grab learner.env.get_current_step()
        '''
        try:
            if type(learner.agents) == list:
                for agent in learner.agents:
                    self._save_agent_state_(learner, agent, checkpoint_num)
            else:
                self._save_agent_state_(learner, learner.agents, checkpoint_num)
        except AttributeError: # when learner has only 1 agent
            try:
                if type(learner.agent) == list:
                    for agent in learner.agent:
                        self._save_agent_state_(learner, agent, checkpoint_num)
                else:
                    self._save_agent_state_(learner, learner.agent, checkpoint_num)
            except AttributeError:
                # self.log(learner)
                assert False, "Couldn't find the Learners agent/s..."

    # def _save_agent(self, learner, agent, checkpoint_num=None):
    #     '''
    #         Mechanics of saving an individual Agent
    #             1-  Pickles the agent object and save attributes
    #             2-  Uses the save() method from the Agent class because Agents could have diff network structures
    #         Input
    #             @learner        Learner who owns the agent
    #             @agent          Agent we want to save
    #     '''
    #     if not self.need_to_save: return
    #     agent_path = self.get_new_agent_dir(learner, agent)
    #     fh.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))
    #     if checkpoint_num is None:
    #         try:
    #             checkpoint_num = learner.env.done_count
    #         except:
    #             checkpoint_num = learner.done_count
    #     agent.save(agent_path, checkpoint_num)

    def _save_agent_state_(self, learner, agent, checkpoint_num=None):
        '''
            Mechanics of saving an individual Agent
                1-  Pickles the agent object and save attributes
                2-  Uses the save() method from the Agent class because Agents could have diff network structures
            Input
                @learner        Learner who owns the agent
                @agent          Agent we want to save
        '''
        if not self.need_to_save: return
        agent_path = self.get_new_agent_dir(learner, agent)
        agent.save_state_dict(agent_path)

    '''
        LOAD METHODS
    '''

    def load_agents(self, path:str, absolute_path=True, load_latest=True, device=torch.device('cpu')) -> list:
        '''
        Loads agents from sratch, initializing new Agent object
        '''
        _path = path
        if load_latest and self.__folder_name__['latest'] not in path:
            _path = os.path.join(path, self.__folder_name__['latest'])
        agents_states = self._load_agents_states(_path, absolute_path=absolute_path, device=device)
        agents = []
        for state_dict in agents_states:
            agent_state_dict = torch.load(state_dict, map_location=device)
            _new_agent = self.__create_agent_from_state_dict__(agent_state_dict)
            agents.append(_new_agent)
        return agents

    def load_agent_of_id(self, path:str, agent_id:int, absolute_path=True, device=torch.device('cpu')):
        '''Loads agent of some ID'''
        agents_states = self._load_agents_states(path, agent_id=agent_id, absolute_path=absolute_path, device=device)
        agents = []
        for state_dict in agents_states:
            agent_state_dict = torch.load(state_dict, map_location=device)
            _new_agent = self.__create_agent_from_state_dict__(agent_state_dict)
            agents.append(_new_agent)
        return agents

    def reload_agents(self, agents:list, path:str, absolute_path=True, load_latest=True, device=torch.device('cpu')) -> list:
        '''
        Loads new agent states
        '''
        if len(agents) == 0 or agents[0] == None:
            '''First time loading... use load_agents directly'''
            return self.load_agents(path, absolute_path=absolute_path, load_latest=load_latest, device=device)
        else:
            _path = path
            if load_latest and self.__folder_name__['latest'] not in path:
                _path = os.path.join(path, self.__folder_name__['latest'])
            for a in agents:
                agents_states = self._load_agents_states(_path, agent_id=a.id, absolute_path=absolute_path, device=device)
                assert len(agents_states) == 1, "mmmmm something weird here because we should find only 1 state dict"
                agent_state_dict = torch.load(agents_states[0], map_location=device)
                a.load_state_dict(agent_state_dict)
            return agents


    def _load_agents_states(self, path, agent_id=None, absolute_path=True, device=torch.device('cpu')) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load all those agents with their corresponding policies
            Input
                @path       Path where the agents files will be located
        '''
        if not absolute_path:
            path = '/'.join([self.base_url, path])
        agents_states = dh.find_pattern_in_path(path, '{id}.state'.format(id=agent_id if agent_id is not None else ''))
        assert len(agents_states) > 0, "No agents found in {} with agent_id {}".format(path, agent_id)
        self.log("Found to load {}".format(agents_states), verbose_level=2)
        return agents_states

    def __create_agent_from_state_dict__(self, state_dict):
        '''Instantiate new agent object and load it's attributes'''
        '''This does the actual mechanics of loading - low level'''
        _new_agent_class = ch.load_class(state_dict['class_module'], state_dict['class_name'])
        _new_agent = _new_agent_class(*state_dict['inits'])
        _new_agent.load_state_dict(state_dict)
        # _new_agent = self.__load_agent_states__(_new_agent, state_dict)
        self.log("Loaded {}".format(str(_new_agent)), to_print=True, verbose_level=2)
        return _new_agent

    def _load_buffer(self, path) -> list:
        '''
            For now, we have only 1 buffer per learner
            Input
                @path       Learner path
        '''

        buffer_pickle = dh.find_pattern_in_path(path, 'buffer_cls.pickle')
        assert len(buffer_pickle) > 0, "No buffer found in {}".format(path)
        self.log("Loading Buffer..", to_print=True)
        self.log("\t{}\n".format(buffer_pickle[0]))
        return fh.load_pickle_obj(buffer_pickle[0])

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Admin']:
            text = '{}\t\t{}'.format(str(self), msg)
            self.logger.info(text, self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<Admin>"

###########################################################################

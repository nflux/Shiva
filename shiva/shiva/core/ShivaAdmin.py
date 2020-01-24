import os
import configparser
import inspect
from tensorboardX import SummaryWriter

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
        'agent':        'A{id}',
    }

    def __init__(self, logger, config=None):
        if config is not None:
            self.init(config)
        self.logger = logger
        self.base_url = os.path.abspath('.')

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

        self._meta_learner_dir = ''
        self._learner_dir = {}
        self._agent_dir = {}
        self.writer = {}

    def __str__(self):
        return "<ShivaAdmin>"
    
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
            dh.make_dir(os.path.join(self.base_url, directory), overwrite=True)

    def add_meta_profile(self, meta_learner, folder_name: str=None, overwrite=False) -> None:
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
        new_dir = dh.make_dir_timestamp(os.path.join(self.base_url, self.runs_url, folder_name), overwrite=overwrite)
        self._meta_learner_dir = new_dir
        self.log("New MetaLearner @ {}".format(self._meta_learner_dir), to_print=True)

    def add_learner_profile(self, learner, function_only=False) -> None:
        '''
            Needed to keep track of the Learner and it's directory.
            Creates a folder for it if save flag is True.

            Input
                @learner            Learner instance to be saved
                @function_only      When we only want to use this functionality without profiling, used for distributed processes but running locally
        '''
        if not self.need_to_save: return
        if learner.id not in self._learner_dir:
            if function_only:
                # need to create a new fake "meta profile" to create root session folder structures for this learner
                folder_name = self.__folder_name__['metalearner'].format(algorithm=learner.configs['Algorithm']['type'], env=learner.configs['Environment']['env_name'])
                self.add_meta_profile(None, folder_name, overwrite=True)
            self._learner_dir[learner.id] = {}
            new_dir = dh.make_dir( os.path.join(self._meta_learner_dir, self.__folder_name__['learner'].format(id=str(learner.id))) )
            self._learner_dir[learner.id]['base'] = new_dir
            self._learner_dir[learner.id]['checkpoint'] = [] # keep track of each checkpoint directory
            temp_dir = dh.make_dir( os.path.join(self._learner_dir[learner.id]['base'], 'temp') )
            self._learner_dir[learner.id]['temp'] = temp_dir
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
            checkpoint_dir = self._learner_dir[learner.id]['temp']
        else:
            checkpoint_dir = dh.make_dir(os.path.join( self._learner_dir[learner.id]['base'], self.__folder_name__['checkpoint'].format(ep_num=str(checkpoint_num)) ))
        self._learner_dir[learner.id]['checkpoint'].append(checkpoint_dir)
        # self._save_learner(learner, checkpoint_num)
        self._save_learner_agents(learner, checkpoint_num)

    def get_last_checkpoint(self, learner):
        if len(self._learner_dir[learner.id]['checkpoint']) == 0:
            # no checkpoint available
            return False
        else:
            return self._learner_dir[learner.id]['checkpoint'][-1]

    def get_temp_directory(self, learner):
        assert learner.id in self._learner_dir, "Learner was not profiled by ShivaAdmin, try calling Admin.add_learner_profile at initialization "
        return self._learner_dir[learner.id]['temp']

    def _add_agent_checkpoint(self, learner, agent):
        '''
            Creates the corresponding folder for the agent checkpoint
            Instantiates the Tensorboard SummaryWriter if doesn't exists

            Input
                @learner            Learner instance ref owner of the Agent
                @agent              Agent instance ref to be saved
        '''
        if not self.need_to_save: return
        new_dir = dh.make_dir( os.path.join( self._learner_dir[learner.id]['checkpoint'][-1], self.__folder_name__['agent'].format(id=str(agent.id)) ) )
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
            return self._learner_dir[learner.id]['temp']
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
            new_dir = dh.make_dir( os.path.join( self._learner_dir[learner.id]['summary'], self.__folder_name__['agent'].format(id=str(agent.id)) ) )
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
        if type(agent) == int:
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

    def _save_learner(self, learner=None) -> None:
        '''
            Mechanics of saving a Learner
                1.  Pickles the Learner class
                2.  Pickles the Buffer
                3.  Saves the Learners config

            Input
                @learner        Learner instance we want to save
        '''
        learner = self.caller if learner is None else learner
        self.add_learner_profile(learner) # will only add if was not profiled before
        # save learner pickle
        learner_data_dir = dh.make_dir( os.path.join(self._learner_dir[learner.id]['checkpoint'][-1], self.__folder_name__['learner_data']) )
        fh.save_pickle_obj(learner, os.path.join(learner_data_dir, 'learner_cls.pickle'))
        # save buffer
        fh.save_pickle_obj(learner.buffer, os.path.join(learner_data_dir, 'buffer_cls.pickle'))
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
                    self._save_agent(learner, agent, checkpoint_num)
            else:
                self._save_agent(learner, learner.agents, checkpoint_num)
        except AttributeError: # when learner has only 1 agent
            try:
                if type(learner.agent) == list:
                    for agent in learner.agent:
                        self._save_agent(learner, agent, checkpoint_num)
                else:
                    self._save_agent(learner, learner.agent, checkpoint_num)
            except AttributeError:
                self.log(learner)
                assert False, "Couldn't find the Learners agent/s..."

    def _save_agent(self, learner, agent, checkpoint_num=None):
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
        fh.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))
        if checkpoint_num is None:
            try:
                checkpoint_num = learner.env.done_count
            except:
                checkpoint_num = learner.done_count
        agent.save(agent_path, checkpoint_num)

    '''
        LOAD METHODS
    '''

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
        assert False, "Not implemented"
        self.log("Loading MetaLearner")

    def _load_learner(self, path: str, includeAgents: bool=True) -> object:
        '''
            DO WE REALLY NEED THIS?

            Input
                @path               Path to the Learner episode
                @includeAgents      Boolean if want to include the agents in the returned Learner

            Returns
                Learner instance
        '''
        assert False, 'Not implemented - Do we really need this method??????'
        # learner_pickle = dh.find_pattern_in_path(path, 'learner_cls.pickle')
        # assert len(learner_pickle) == 1, "{} learner pickles were found in {}".format(str(len(learner_pickle)), path)
        # learner_pickle = learner_pickle[0]
        # self.log('Loading Learner\n\t{}\n'.format(learner_pickle))
        # learner_path, _ = os.path.split(learner_pickle)
        # _new_learner = fh.load_pickle_obj(learner_pickle)
        # if includeAgents:
        #     _new_learner.agents = self._load_agents(learner_path)
        #     _new_learner.buffer = self._load_buffer(learner_path)
        # return _new_learner

    def _load_agents(self, path, absolute_path=True) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load all those agents with their corresponding policies

            Input
                @path       Path where the agents files will be located
        '''
        if not absolute_path:
            path = '/'.join([self.base_url, path])
        agents = []
        agents_pickles = dh.find_pattern_in_path(path, 'agent_cls.pickle')
        agents_policies = dh.find_pattern_in_path(path, '.pth')
        assert len(agents_pickles) > 0, "No agents found in {}".format(path)
        # self.log("Loading Agents..")
        for agent_pickle in agents_pickles:
            _new_agent = fh.load_pickle_obj(agent_pickle)
            this_agent_folder = agent_pickle.replace('agent_cls.pickle', '')
            this_agent_policies = []
            # find this agents corresponding policies
            for pols in agents_policies:
                if this_agent_folder in pols:
                    this_agent_policies.append(pols)
                    policy_name = pols.replace(this_agent_folder, '').replace('.pth', '')
                    _new_agent.load_net(policy_name, pols)
            self.log("Loading Agent {} with {} networks".format(agent_pickle.replace(os.getcwd(), ''), len(this_agent_policies)), to_print=True)
            agents.append(_new_agent)
        return agents

    def _load_agent(self, path) -> list:
        '''
            For a given @path, the procedure will walk recursively over all the folders inside the @path
            And find all the agent_cls.pickle and policy.pth files to load a SINGLE AGENT

            InputW
                @path       Path where the agents files will be located
        '''
        agent_pickle = dh.find_pattern_in_path(path, 'agent_cls.pickle')
        agent_policy = dh.find_pattern_in_path(path, '.pth')
        assert len(agent_pickle) > 0, "No agent found in {}".format(path)
        assert len(agent_pickle) == 1, "Multiple agent_cls.pickles found. Try using shiva._load_agents()"
        agent_pickle = agent_pickle[0]
        self.log("Loading Agent..\n\t{} with {} networks\n".format(agent_pickle, len(agent_policy)), to_print=True)
        _new_agent = fh.load_pickle_obj(agent_pickle)
        this_agent_folder = agent_pickle.replace('agent_cls.pickle', '')
        for pols in agent_policy:
            if this_agent_folder in pols:
                # this_agent_policies.append(pols)
                policy_name = pols.replace(this_agent_folder, '').replace('.pth', '')
                _new_agent.load_net(policy_name, pols)
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

    def log(self, msg, to_print=False):
        text = "Admin\t\t{}".format(msg)
        self.logger.info(text, to_print)
    

###########################################################################
#         
###########################################################################
import sys, time
import numpy as np
from mpi4py import MPI

from shiva.learners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_config_file_2_dict, merge_dicts
from shiva.helpers.utils.Tags import Tags

from typing import List, Dict, Tuple, Any, Union

class MPIPBTMetaLearner(MetaLearner):

    def __init__(self, configs: Dict[str, Any]):
        """
        MetaLearner implementation for a distributed architecture where we can optionally enable Population Based Training meta learning.
        For usage details go to the MPIPBTMetaLearner config templates and explanations.

        Args:
            configs (Dict[str, Any]): config used for the run
        """
        # for future MPI abstraction
        self.id = 0
        self.info = MPI.Status()
        super(MPIPBTMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self._preprocess_config()
        self._launch()

    def _launch(self) -> None:
        """
        Launches Shiva components in their own MPI process:
        * Administrative: IOHandler
        * Training components: MPIMultiEnv, MPILearner
        * Evaluation components: MPIMultiEvaluationWrapper (when PBT is enabled)

        And then the infinite loop under the `run`_ method.

        Returns:
            None

        """
        self._launch_io_handler()
        self._launch_menvs()
        self._launch_learners()
        self._launch_mevals()
        self.run()

    def run(self) -> None:
        """
        Infinite loop for the high level operations for meta learning purposes where we interface between the Learner and Evaluation when PBT is enabled.

        Returns:
            None
        """
        self.start_evals()
        self.review_training_matches() # send first match
        self.is_running = True
        while self.is_running:
            time.sleep(self.configs['Admin']['time_sleep']['MetaLearner'])
            self.get_multieval_metrics()
            # self.review_training_matches() # dont send new training matches during runtime - keep fixed for now
            self.evolve()
            self.check_states()
        self.close()

    def check_states(self) -> None:
        """
        This function would check the state of all the components. Might need some work, although for now checks if one of the Learners has closed and sets the `is_running`_ flag accordingly.

        Returns:
            None
        """
        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.close, status=self.info):
            # one of the Learners run the close()
            learner_spec = self.learners.recv(None, source=self.info.Get_source(), tag=Tags.close)
            self.log("Learner {} CLOSED".format(learner_spec['id']), verbose_level=2)
            # used only to stop the whole session, for running profiling experiments..
            self.is_running = False

    def close(self) -> None:
        """
        Function to be called when a process withing the pipeline has crashed or closed. Currently, we only check state for the Learners.
        This will send a message to all existing processes and tell them to terminate.

        Returns:
            None
        """
        self.log("Started closing", verbose_level=1)
        '''Send message to childrens'''
        for i in range(self.num_menvs):
            self.menvs.send(True, dest=i, tag=Tags.close)
        for i in range(self.num_learners):
            self.learners.send(True, dest=i, tag=Tags.close)
        self.io.send(True, dest=0, tag=Tags.close)

        self.menvs.Disconnect()
        self.log("Closed MultiEnvs", verbose_level=2)
        self.learners.Disconnect()
        self.log("Closed Learners", verbose_level=2)
        if self.pbt:
            self.mevals.Disconnect()
            self.log("Closed MultiEvals", verbose_level=1)
        self.io.Disconnect()
        self.log("FULLY CLOSED", verbose_level=1)

    def start_evals(self) -> None:
        """
        Gives the Start Flag to the Evaluation processes to start collecting metrics. Evaluation pipeline won't start until this flag is received.

        Returns:
            None
        """
        if self.pbt:
            self.mevals.bcast(True, root=MPI.ROOT)

    def evolve(self):
        """
        Executes the high level operations to create a new evolution config to be handled to the Learners who request an evolution config.
        It will proceed only if: PBT is enabled, a Learner has requested a config and we have received at least 1 Ranking from the Evaluation pipeline.
        If so, will create a evolution config for the Learner and send it. It will never sent a Evolution config to a Learner twice if we haven't received new Rankings between both requests.

        Returns:
            None
        """
        if self.pbt and self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution_request, status=self.info):

            learner_spec = self.learners.recv(None, source=self.info.Get_source(), tag=Tags.evolution_request)
            assert learner_spec['id'] == self.info.Get_source(), "mm - just checking :)"

            if hasattr(self, 'rankings') and not self.evols_sent[learner_spec['id']]:
                roles_evo = self._get_evolution_config(learner_spec)
                self.learners.send(roles_evo, dest=self.info.Get_source(), tag=Tags.evolution_config)
                self.log("Evo Sent {}".format(roles_evo), verbose_level=2)
                self.evols_sent[learner_spec['id']] = True
            else:
                pass
                # self.log("Don't send to Learner duplicate ranking", verbose_level=2)

    def _get_evolution_config(self, learner_spec: Dict[str, Any]) -> Dict[str, Union[int, str]]:
        """
        Creates a new Evolution config for the given Learner spec. For more information about `learner_spec` go to `MPILearner._get_learner_specs()` function.
        This function evaluates the current Ranking for all the Agents that this Learner owns and detemines the evolution procedures that the Learner must follow.
        For elite agents, no evolution is needed.
        For mid ranking agents, they will do `t_test` comparison with some other elite/mid agent to see if they truncate or not. Then they will perturb or resample hyperparameters.
        For low ranking agents, they will truncate from elite agents and perturb or resample hyperparameters.

        Args:
            learner_spec (Dict[str, Any]): the Learner spec for who we need to create a evolution config

        Returns:
            Evolution config (Dict[str, Union[int, str]])
        """
        roles_evo = []
        for role, agent_ids in learner_spec['role2ids'].items():
            agent_evo = dict()
            '''@agent_ids is a list of a single agent'''
            '''Assuming that each Learner has 1 agent per role'''
            agent_id = agent_ids[0]
            # ranking = np.where(self.rankings[role] == agent_id)[0]
            ranking = self.rankings[role].index(agent_id)
            agent_evo['agent_id'] = agent_id
            agent_evo['ranking'] = ranking
            if ranking <= self.top_20[role]:
                agent_evo['msg'] = 'You are good'
                agent_evo['evolution'] = False
            elif self.top_20[role] < ranking < self.bottom_20[role]:
                agent_evo['msg'] = 'Middle of the Pack'
                agent_evo['evolution'] = True
                agent_evo['evo_agent_id'] = self.rankings[role][np.random.choice(range(self.bottom_20[role]))]
                agent_evo['evo_path'] = self.get_learner_spec(agent_evo['evo_agent_id'])['load_path']
                # agent_evo['evo_ranking']= np.where(self.rankings[role] == agent_evo['evo_agent_id'])
                agent_evo['evo_ranking'] = self.rankings[role].index(agent_evo['evo_agent_id'])
                agent_evo['exploitation'] = 't_test'
                agent_evo['exploration'] = np.random.choice(['perturb', 'resample'])
            else:
                agent_evo['msg'] = 'You suck'
                agent_evo['evolution'] = True
                agent_evo['evo_agent_id'] = self.rankings[role][ np.random.choice(range(self.top_20[role])) if self.top_20[role] != 0 else 0 ]
                agent_evo['evo_path'] = self.get_learner_spec(agent_evo['evo_agent_id'])['load_path']
                # agent_evo['evo_ranking'] = np.where(self.rankings[role] == agent_evo['evo_agent_id'])
                agent_evo['evo_ranking'] = self.rankings[role].index(agent_evo['evo_agent_id'])
                agent_evo['exploitation'] = 'truncation'
                agent_evo['exploration'] = np.random.choice(['perturb', 'resample'])
            roles_evo.append(agent_evo)
        return roles_evo

    def get_multieval_metrics(self) -> None:
        """
        Receive new rankings from MPIMultiEvaluationWrapper

        Returns:
            None
        """
        while self.mevals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.rankings, status=self.info):
            self.rankings = self.mevals.recv(None, source=self.info.Get_source(), tag=Tags.rankings)
            self.evols_sent = {i:False for i in range(self.num_learners)}
            self.log('Got New Rankings {}'.format(self.rankings), verbose_level=1)

    def review_training_matches(self):
        """
        Creates new pairs of training matches and sends them to the MultiEnvs to load.
        It's used during initialization but not being used during runtime. Would be better to have a match making process using ELO score.

        Returns:
            None
        """
        self.current_matches = self.get_training_matches()
        for ix, match in enumerate(self.current_matches):
            self.menvs.send(match, dest=ix, tag=Tags.new_agents)

    def get_training_matches(self) -> List[Dict[str, Dict]]:
        """
        Creates unique pairs of training matches for the currently running MultiEnvs.

        Returns:
            List[Dict[str, Any]: list of training matches. Each training match is a Dict that maps role names to Learner specs.
        """
        matches = []
        self.log(self.learners_specs)
        for i in range(0, self.num_learners, self.num_learners_per_map):
            low = i
            high = i + self.num_learners_per_map
            m = {}
            for l_ix in range(low, high, 1):
                learner_spec = self.learners_specs[l_ix]
                for role in learner_spec['roles']:
                    m[role] = learner_spec
            # test all roles are filled
            for role in self.roles:
                assert role in m, "Role {} not found on this created match {}".format(role, m)
            matches += [m] * self.num_menvs_per_learners_map

        assert len(matches) == self.num_menvs, "Tried to create unique matches for MultiEnvs so that all Learners are training. " \
                                               f"Created the wrong number of matches: we have {self.num_envs} MultiEnvs and created {len(matches)} matches"
        # log match to some other format
        matches_log = []
        for ix, m in enumerate(matches):
            match_log = "Match {}: ".format(ix)
            for role, l_spec in m.items():
                match_log += " {} to Learner {} |".format(role, l_spec['id'])
            matches_log.append(match_log)
        self.log("Created {} new training matches\n{}".format(len(matches), "\n".join(matches_log)), verbose_level=2)
        return matches

    def has_pair(self, agent_id: int, role: str) -> Tuple[bool, List[int]]:
        """
        Check if the given Agent ID and Role has an exclusive that needs to be paired with. This is the case when one of the Agents is under a centralized critic.
        This function could be potentially used for pair matching. Not currently being used.

        Args:
            agent_id (int): Agent ID for which we want to check if it has a pair
            role (str): role name for the agent who we are trying to find a pair

        Returns:
            Tuple[bool, List[int]]: bool indicates if a pair was found. If a pair was found the second argument contains a list of the Agent IDs that are possible pairs.
        """
        has = False
        for l_spec in self.learners_specs:
            if agent_id in l_spec['agent_ids'] and len(l_spec['agent_ids']) > 1:
                for l_roles in l_spec['roles']:
                    if role != l_roles:
                        has = True
                if has:
                    return True, l_spec['role2ids']
        return False, []

    def get_role(self, agent_id: int) -> str:
        """
        This function returns the role name for a fiven Agent ID.

        Args:
            agent_id (int): Agent ID for who we are trying to find the role name.

        Returns:
            str: role name
        """
        for role, role_agent_ids in self.role2ids.items():
            if agent_id in role_agent_ids:
                return role

    def get_learner_id(self, agent_id: int) -> int:
        """
        This function returns the Learner ID who owns the given Agent ID.

        Args:
            agent_id (int): Agent ID for who we are trying to get the Learner ID

        Returns:
            int: Learner ID that owns the given Agent ID
        """
        for spec in self.learners_specs:
            if agent_id in spec['agent_ids']:
                return spec['id']

    def get_learner_spec(self, agent_id: int) -> Dict[str, Any]:
        """
        Similar to `get_learner_id`_ but instead returns the Learner Spec. To learn more about the Learner Spec go to `Learner.get_learner_specs`.

        Args:
            agent_id (int): Agent ID for who are trying to get the Learner Spec

        Returns:
            Dict[str, Any]: for more details about this Spec go to `Learner.get_learner_specs`
        """
        for spec in self.learners_specs:
            if agent_id in spec['agent_ids']:
                return spec

    def _launch_io_handler(self) -> None:
        """
        Spawns the process for the IOHandler.

        Returns:
            None
        """
        args = ['shiva/core/IOHandler.py', '-a', self.configs['Admin']['iohandler_address']]
        self.io = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=1)
        self.io.bcast(self.configs, root=MPI.ROOT)

    def _launch_menvs(self) -> None:
        """
        Spawns the processes for the MPIMultiEnv.

        Returns:
            None
        """
        self.num_menvs = self.num_learners_maps * self.num_menvs_per_learners_map # in order to have all Learners interact with at least 1 MultiEnv
        # self.configs['Environment']['menvs_io_port'] = self.configs['IOHandler']['menvs_io_port']
        self.menvs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(str(len(menvs_specs)), str(len(menvs_specs[0].keys()))), verbose_level=1)
        self.configs['MultiEnv'] = menvs_specs

    def _launch_mevals(self) -> None:
        """
        Spawns the processes for the MPIMultiEvaluationWrapper.

        Returns:
            None
        """
        self.configs['Evaluation']['agent_ids'] = self.agent_ids
        self.configs['Evaluation']['roles'] = self.roles if hasattr(self, 'roles') else []
        self.configs['Evaluation']['role2ids'] = self.role2ids
        self.configs['Evaluation']['learners_specs'] = self.learners_specs
        # self.configs['Evaluation']['evals_io_port'] = self.configs['IOHandler']['evals_io_port']

        if self.pbt:
            self.rankings_size = {role:len(self.role2ids[role]) for role in self.roles} # total population for each role
            self.top_20 = {role:int(self.rankings_size[role] * self.configs['Evaluation']['expert_population']) for role in self.roles}
            self.bottom_20 = {role:int(self.rankings_size[role] * (1-self.configs['Evaluation']['expert_population'])) for role in self.roles}
            self.evols_sent = {l_spec['id']:False for l_spec in self.learners_specs}

            self.mevals = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIMultiEvaluationWrapper.py'], maxprocs=self.num_mevals)
            self.mevals.bcast(self.configs, root=MPI.ROOT)
            mevals_specs = self.mevals.gather(None, root=MPI.ROOT)
            self.configs['MultiEvals'] = mevals_specs
        else:
            '''Overwrite evolution function to do nothing'''
            self.get_multieval_metrics = lambda: None
            self.evolve = lambda: None

    def _launch_learners(self) -> None:
        """
        Spawns the processes for the MPILearner. Does some checking on the roles being assigned to each Learner so that at least one Role on the Environment is being occupied by a Learner.

        Returns:
            None
        """
        self.learners_configs = self.generate_learners_configs()
        self.learners = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        self.learners.scatter(self.learners_configs, root=MPI.ROOT)
        self.learners_specs = self.learners.gather(None, root=MPI.ROOT)

        '''Review the Population for each Role'''
        self.agent_ids = []
        self.agent_roles = [] # same order as IDs
        self.role2ids = {role:[] for role in self.roles}
        for spec in self.learners_specs:
            '''Double check that there are no duplicate agent IDs across Learners'''
            for id in spec['agent_ids']:
                assert id not in self.agent_ids, "Duplicate agent ID in between Learners - this might break Evaluation"
                self.agent_ids.append(id)
            '''Collect the total agent IDs we have for each Role'''
            for role, ids in spec['role2ids'].items():
                self.role2ids[role] += ids
        self.agent_ids = np.array(self.agent_ids)
        self.configs['Learners_Specs'] = self.learners_specs
        self.log("Got {} LearnerSpecs".format(len(self.learners_specs)), verbose_level=1)

    def generate_learners_configs(self) -> List[Dict[str, Dict[str, Union[List, Dict]]]]:
        """
        Generates the Learners configs by using the Learners Map in the main config file. Check that the Learners assignment with the environment Roles are correct. Does some default assignments as well.

        Returns:
            List[Dict[str, Dict[str, Union[List, Dict]]]]: A list of config dictionaries.
        """
        self.learner_configs = []
        if hasattr(self, 'learners_map'):
            self.roles = self.configs['MultiEnv'][0]['env_specs']['roles']
            '''First check that all Agents Roles are assigned to a Learner'''
            for ix, roles in enumerate(self.roles):
                if roles in set(self.learners_map.keys()):
                    pass
                else:
                    assert "Agent Roles {} is not being assigned to any Learner\nUse the 'learners_map' attribute on the [MetaLearner] section".format(roles)
            '''Do some preprocessing before spreading the config'''
            # load each one of the configs and keeping same order
            self.learners_configs = []
            for learner_ix, (config_path, learner_roles) in enumerate(self.learners_map.items()):
                learner_config = load_config_file_2_dict(config_path)
                learner_config = merge_dicts(self.configs, learner_config)
                self.log("Learner {} with roles {}".format(len(self.learners_configs), learner_roles), verbose_level=2)
                learner_config['Learner']['roles'] = learner_roles
                learner_config['Learner']['pbt'] = self.pbt
                learner_config['Algorithm']['manual_seed'] = self.manual_seed + learner_ix if 'manual_seed' not in learner_config['Algorithm'] else learner_config['Algorithm']['manual_seed']
                learner_config['Agent']['manual_seed'] = learner_config['Algorithm']['manual_seed'] if 'manual_seed' not in learner_config['Agent'] else learner_config['Agent']['manual_seed']
                learner_config['Agent']['pbt'] = self.pbt
                self.learners_configs.append(learner_config)
            self.learners_configs = self.learners_configs * self.num_learners_maps
        elif 'Learner' in self.configs:
            self.learners_configs = [self.configs.copy() for _ in range(self.num_learners)]
        else:
            assert "Error processing Learners Configs"
        return self.learners_configs

    def _preprocess_config(self) -> None:
        """
        Do some preprocessing and injecting default values in the main config.

        Returns:
            None
        """
        assert hasattr(self, 'learners_map'), "Need 'learners_map' attribute on the [MetaLearner] section of the config"

        # calculate number of learners using the learners_map dict
        self.num_menvs_per_learners_map = self.num_menvs_per_learners_map if hasattr(self, 'num_menvs_per_learners_map') else 1
        self.num_learners_maps = self.num_learners_maps if hasattr(self, 'num_learners_maps') else 1
        self.num_learners_per_map = len(self.learners_map.keys())
        self.num_learners = self.num_learners_per_map * self.num_learners_maps
        self.configs['MetaLearner']['num_learners'] = self.num_learners
        self.log("Preprocess config\nOriginal LearnerMap {}\nNum Learners\t{}\nNum Learners Per Map\t{}\nNum Learners Maps\t{}\nNum MultiEnvs Per Learner Map\t{}".format(self.learners_map, self.num_learners, self.num_learners_per_map, self.num_learners_maps, self.num_menvs_per_learners_map), verbose_level=3)

        self.configs['Evaluation']['manual_seed'] = self.manual_seed if 'manual_seed' not in self.configs['Evaluation'] else self.configs['Evaluation']['manual_seed']
        self.configs['Environment']['manual_seed'] = self.manual_seed if 'manual_seed' not in self.configs['Environment'] else self.configs['Environment']['manual_seed']

    def __str__(self):
        return "<Meta(id={})>".format(self.id)
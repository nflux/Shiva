import sys, time
import logging
import numpy as np
from mpi4py import MPI

from shiva.core.admin import Admin, logger
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_config_file_2_dict, merge_dicts
from shiva.helpers.misc import terminate_process
from shiva.utils.Tags import Tags

class MPIPBTMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIPBTMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self._preprocess_config()
        self.info = MPI.Status()
        self.launch()

    def launch(self):
        self._launch_io_handler()
        self._launch_menvs()
        self._launch_learners()
        self._launch_mevals()

        if hasattr(self, 'learners_map'):
            self.rankings_size = {role:len(self.role2ids[role]) for role in self.roles} # total population for each role
            self.bottom_20 = {role:int(self.rankings_size[role] * .50) for role in self.roles}
            self.top_20 = {role:int(self.rankings_size[role] * .50) for role in self.roles}
            self.evolve = self._roles_evolve
            self.evols_sent = {l_spec['id']:False for l_spec in self.learners_specs}
        else:
            self.rankings_size = self.configs['Evaluation']['num_agents']
            self.bottom_20 = int(self.rankings_size * .80)
            self.top_20 = int(self.rankings_size * .20)
            self.evolve = self._single_agent_evolve
        self.run()

    def run(self):
        while True:
            time.sleep(0.001)
            self.get_multieval_metrics()
            self.evolve()

    def _roles_evolve(self):
        '''Evolve only if we received Rankings'''
        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution, status=self.info):
            learner_spec = self.learners.recv(None, source=self.info.Get_source(), tag=Tags.evolution)
            if hasattr(self, 'rankings'): #self.evols_sent[learner_spec['id']]:
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
                    self.log("id {} r {} , top_20 {}, bottom_20 {}".format(agent_ids, ranking, self.top_20[role], self.bottom_20[role]))
                    if ranking <= self.top_20[role]:
                        # Good agent
                        agent_evo['evolution'] = False
                    elif self.top_20[role]  < ranking < self.bottom_20[role]:
                        # Middle of the Pack
                        agent_evo['evolution'] = True
                        agent_evo['evo_agent'] = self.rankings[role][ np.random.choice(range(self.bottom_20[role])) ]
                        # agent_evo['evo_ranking']= np.where(self.rankings[role] == agent_evo['evo_agent'])
                        agent_evo['evo_ranking'] = self.rankings[role].index(agent_evo['evo_agent'])
                        agent_evo['exploitation'] = 't_test'
                        agent_evo['exploration'] = np.random.choice(['perturb', 'resample'])
                    else:
                        # You suck
                        agent_evo['evolution'] = True
                        if not self.top_20[role] == 0:
                            agent_evo['evo_agent'] = self.rankings[role][ np.random.choice(range(self.top_20[role])) ]
                        else:
                            agent_evo['evo_agent'] = self.rankings[role][0]
                        # agent_evo['evo_ranking'] = np.where(self.rankings[role] == agent_evo['evo_agent'])
                        agent_evo['evo_ranking'] = self.rankings[role].index(agent_evo['evo_agent'])
                        agent_evo['exploitation'] = 'truncation'
                        agent_evo['exploration'] = np.random.choice(['perturb', 'resample'])
                    roles_evo.append(agent_evo)

                self.log("Sending Evo {}".format(roles_evo))
                self.learners.send(roles_evo, dest=self.info.Get_source(), tag=Tags.evolution_config)
                # self.evols_sent[learner_spec['id']] = True
                delattr(self, 'rankings')

    def _single_agent_evolve(self):
        if hasattr(self, 'rankings') and self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution):
            self.log('MetaLearner Received evolution request')
            info = MPI.Status()
            agent_nums = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.evolution, status=self.info)  # block statement
            self.log("After getting the agent nums")
            learner_source = self.info.Get_source()
            for agent_id in agent_nums:
                evo = dict()
                ranking = np.where(self.rankings == agent_id)[0]
                print('Current Agent Ranking: ', ranking)
                evo['agent_id'] = evo['agent'] = agent_id
                evo['ranking'] = ranking
                if ranking <= self.top_20:
                    evo['evolution'] = False
                    print('Do Not Evolve')
                elif self.top_20  < ranking < self.bottom_20:
                    print('Middle of the Pack: {}'.format(learner_source))
                    evo['evolution'] = True
                    evo['evo_agent'] = self.rankings[np.random.choice(range(self.bottom_20))]
                    evo['evo_ranking']= np.where(self.rankings == evo['evo_agent'])
                    evo['exploitation'] = 't_test'
                    evo['exploration'] = np.random.choice(['perturb', 'resample'])
                else:
                    print('You suck')
                    evo['evolution'] = True
                    if not self.top_20 == 0:
                        evo['evo_agent'] = self.rankings[np.random.choice(range(self.top_20))]
                    else:
                        evo['evo_agent'] = self.rankings[0]
                    evo['evo_ranking'] = np.where(self.rankings == evo['evo_agent'])
                    evo['exploitation'] = 'truncation'
                    evo['exploration'] = np.random.choice(['perturb', 'resample'])
                self.learners.send(evo, dest=learner_source, tag=Tags.evolution_config)
            print('MetaLearner Responded to evolution request with evolution config')
            delattr(self, 'rankings')

    def get_multieval_metrics(self):
        if self.mevals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.rankings, status=self.info):
            self.rankings = self.mevals.recv(None, source=self.info.Get_source(), tag=Tags.rankings)
            self.evols_sent = {i:False for i in range(self.num_learners)}
            self.log('Got Tags.rankings {}'.format(self.rankings))

    def _launch_io_handler(self):
        self.io = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/helpers/io_handler.py'], maxprocs=1)
        self.io.send(self.configs, dest=0, tag=Tags.configs)
        self.io_specs = self.io.recv(None, source=0, tag=Tags.io_config)
        if 'Learner' in self.configs:
            self.configs['Environment']['menvs_io_port'] = self.io_specs['menvs_port']
            self.configs['Learner']['learners_io_port'] = self.io_specs['learners_port']
            # self.configs['Evaluation']['evals_io_port'] = self.io_specs['evals_port']
        else:
            '''With the learners_map, self.configs['Learner'] dict doesn't exist yet - will try some approaches here'''
            self.configs['IOHandler'] = {}
            self.configs['IOHandler']['menvs_io_port'] = self.io_specs['menvs_port']
            self.configs['IOHandler']['learners_io_port'] = self.io_specs['learners_port']
            self.configs['IOHandler']['evals_io_port'] = self.io_specs['evals_port']

    def _launch_menvs(self):
        self.menvs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.configs['Environment']['menvs_io_port'] = self.configs['IOHandler']['menvs_io_port']
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(str(len(menvs_specs)), str(len(menvs_specs[0].keys()))))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_mevals(self):
        self.mevals = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIMultiEvaluationWrapper.py'], maxprocs=self.num_mevals)
        self.configs['Evaluation']['agent_ids'] = self.agent_ids
        self.configs['Evaluation']['roles'] = self.roles if hasattr(self, 'roles') else []
        self.configs['Evaluation']['role2ids'] = self.role2ids
        self.configs['Evaluation']['learner_specs'] = self.learners_specs
        self.configs['Evaluation']['evals_io_port'] = self.configs['IOHandler']['evals_io_port']
        self.mevals.bcast(self.configs, root=MPI.ROOT)
        mevals_specs = self.mevals.gather(None, root=MPI.ROOT)
        self.configs['MultiEvals'] = mevals_specs

    def _launch_learners(self):
        self.learners_configs = self._get_learners_configs()
        self.learners = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        self.learners.scatter(self.learners_configs, root=MPI.ROOT)
        self.learners_specs = self.learners.gather(None, root=MPI.ROOT)

        self.agent_ids = []
        self.agent_roles = [] # same order as IDs
        self.role2ids = {role:[] for role in self.roles}
        for spec in self.learners_specs:
            '''Double checkt that there are no duplicate agent IDs across Learners'''
            for id in spec['agent_ids']:
                assert id not in self.agent_ids, "Duplicate agent ID in between Learners - this might break Evaluation"
                self.agent_ids.append(id)
            '''Collect the total agent IDs we have for each Role'''
            for role, ids in spec['role2ids'].items():
                self.role2ids[role] += ids
        self.agent_ids = np.array(self.agent_ids)
        self.configs['Learners_Specs'] = self.learners_specs
        self.log("Got {} LearnerSpecs".format(len(self.learners_specs)))

    def _get_learners_configs(self):
        '''
            Check that the Learners assignment with the environment Roles are correct
            This will only run if the learners_map is set
        '''
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
            for config_path, learner_roles in self.learners_map.items():
                learner_config = load_config_file_2_dict(config_path)
                learner_config = merge_dicts(self.configs, learner_config)
                learner_config['Learner']['roles'] = learner_roles
                learner_config['Learner']['learners_io_port'] = self.configs['IOHandler']['learners_io_port']
                learner_config['Learner']['pbt'] = self.pbt
                self.learners_configs.append(learner_config)
            self.learners_configs = self.learners_configs * self.num_learners_maps
        elif 'Learner' in self.configs:
            self.learners_configs = [self.configs.copy() for _ in range(self.num_learners)]
        else:
            assert "Error processing Learners Configs"
        return self.learners_configs

    def _preprocess_config(self):
        if hasattr(self, 'learners_map'):
            # calculate number of learners using the learners_map dict
            self.num_learners_maps = self.num_learners_maps if hasattr(self, 'num_learners_maps') else 1
            self.num_learners = len(set(self.learners_map.keys())) * self.num_learners_maps
            self.configs['MetaLearner']['num_learners'] = self.num_learners
        else:
            # num_learners is explicitely in the config
            pass

    def log(self, msg, to_print=False):
        text = "{}\t\t{}".format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<Meta(id=0)>"

    def close(self):
        '''Send message to childrens'''
        self.menvs.Disconnect()
        self.learners.Disconnect()
        self.mevals.Disconnect()
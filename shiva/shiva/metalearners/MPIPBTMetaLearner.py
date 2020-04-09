import sys, time
import logging
import numpy as np
from mpi4py import MPI

from shiva.core.admin import Admin, logger
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_config_file_2_dict, merge_dicts
from shiva.helpers.misc import terminate_process
from shiva.utils.Tags import Tags

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("shiva")

class MPIPBTMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(MPIPBTMetaLearner, self).__init__(configs, profile=False)
        self.configs = configs
        self._preprocess_config()
        self.learner_specs = None
        self.info = MPI.Status()
        self.learner_ids = list()
        #Admin.init(self.configs['Admin'])
        self.launch()



    def launch(self):
        self._launch_io_handler()
        self._launch_menvs()
        self._launch_learners()
        self._launch_mevals()
        self.rankings_size = self.configs['Evaluation']['num_agents']
        self.bottom = int(self.rankings_size * (1.0 - self.configs['Evaluation']['bottom_population_size']))
        self.elite = int(self.rankings_size * self.configs['Evaluation']['expert_population_size'])
        self.run()

    def run(self):

        while True:
            time.sleep(0.00001)
            self.receive_rankings()
            self.process_evolution()
        self.close()


            # self.debug("Got Learners metrics {}".format(learner_specs))

    def _launch_io_handler(self):
        self.io = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/helpers/io_handler.py'], maxprocs=1)
        self.io.send(self.configs,dest=0,tag=Tags.configs)
        self.io_specs = self.io.recv(None, source = 0, tag=Tags.io_config)
        self.configs['Environment']['menvs_io_port'] = self.io_specs['menvs_port']
        self.configs['Learner']['learners_io_port'] = self.io_specs['learners_port']
        self.configs['Evaluation']['evals_io_port'] = self.io_specs['evals_port']


    def _launch_menvs(self):
        self.menvs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/envs/MPIMultiEnv.py'], maxprocs=self.num_menvs)
        self.menvs.bcast(self.configs, root=MPI.ROOT)
        menvs_specs = self.menvs.gather(None, root=MPI.ROOT)
        self.log("Got total of {} MultiEnvsSpecs with {} keys".format(str(len(menvs_specs)), str(len(menvs_specs[0].keys()))))
        self.configs['MultiEnv'] = menvs_specs

    def _launch_learners(self):
        self.learners_configs = self._get_learners_configs()
        self.learners = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/learners/MPILearner.py'], maxprocs=self.num_learners)
        self.learners.scatter(self.learners_configs, root=MPI.ROOT)
        self.learners_specs = self.learners.gather(None, root=MPI.ROOT)
        self.learners.bcast(self.learners_specs, root=MPI.ROOT)
        self.agent_ids = self.learners.gather(None, root=MPI.ROOT)
        self.agent_ids = np.array(self.agent_ids).squeeze(axis=1)
        self.log("AFTER LAUNCHING ALL THE LEARNERS {}".format(self.agent_ids))
        # print("META: This is the specs", learners_specs)
        self.log("Got {}".format(len(self.learners_specs)))
        self.log('Learners Specs: {}'.format(self.learners_specs))

    def _launch_mevals(self):
        self.mevals = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIMultiEvaluationWrapper.py'], maxprocs=self.num_mevals)
        self.mevals.bcast(self.configs, root=MPI.ROOT)
        self.mevals.bcast(self.learners_specs,root=MPI.ROOT)
        mevals_specs = self.mevals.gather(None, root=MPI.ROOT)
        self.mevals.bcast(self.agent_ids,root=MPI.ROOT)
        self.log("AFTER LAUNCHING ALL THE EVALS {}".format(self.agent_ids))
        # self.log("Got total of {} MultiEvalSpecs with {} keys".format(len(mevals_specs), len(mevals_specs[0].keys())))
        self.configs['MultiEvals'] = mevals_specs

    def _get_learners_configs(self):
        '''
            Check that the Learners assignment with the environment Group Names are correct
            This will only run if the learners_map is set
        '''
        self.learner_configs = []
        if hasattr(self, 'learners_map'):
            for ix, agent_group in enumerate(self.configs['MultiEnv'][0]['env_specs']['agents_group']):
                if agent_group in set(self.learners_map.keys()):
                    pass
                else:
                    assert "Config error - Agent Group {} was not found on the [MetaLearner] learners_map attribute".format(agent_group)
            '''Do some preprocessing before spreading the config'''
            # load each one of the configs and keeping same order
            self.learners_configs = []
            for config_path in list(self.learners_map.keys()):
                learner_config = load_config_file_2_dict(config_path)
                learner_config['num_learners'] = self.num_learners
                self.learners_configs.append(merge_dicts(self.configs, learner_config))
        else:
            '''This happens when all configs are in 1 file'''
            self.learners_configs = [self.configs.copy() for _ in range(self.num_learners)]
        return self.learners_configs

    def _preprocess_config(self):
        if hasattr(self, 'learners_map'):
            # calculate number of learners using the learners_map dict
            self.num_learners = len(set(self.learners_map.keys()))
            print("PREPROCESS CONFIG {}".format(self.num_learners) )
            self.configs['MetaLearner']['num_learners'] = self.num_learners
        else:
            # num_learners is explicitely in the config
            pass

    def receive_rankings(self):
        if self.mevals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.rankings):
            self.rankings = self.mevals.recv(None,source=MPI.ANY_SOURCE, tag=Tags.rankings)
            self.log('MetaLearner Rankings: {}'.format(self.rankings))

    def process_evolution(self):
        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution) and hasattr(self, 'rankings'):
            self.log('MetaLearner Received evolution request')
            info = MPI.Status()
            agent_nums = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.evolution,status=self.info)  # block statement
            self.log("After getting the agent nums")
            learner_source = self.info.Get_source()
            for agent_id in agent_nums:
                evo = dict()
                print('Current Agent Ranking: ', np.where(self.rankings == agent_id)[0])
                ranking = np.where(self.rankings == agent_id)[0]
                evo['agent_id'] = agent_id
                if ranking < self.elite:
                    evo['evolution'] = False
                    print('Do Not Evolve')
                    self.learners.send(evo,dest=learner_source,tag=Tags.evolution_config)
                elif self.elite  <= ranking < self.bottom:
                    print('Middle of the Pack: {}'.format(learner_source))
                    evo['evolution'] = True
                    evo['agent'] = agent_id
                    evo['ranking'] = ranking
                    #evo['evo_agent'] = self.rankings[np.random.choice(range(self.bottom))]
                    evo['evo_agent'] = agent_id
                    evo['evo_ranking']= np.where(self.rankings == evo['evo_agent'])
                    evo['exploitation'] = 't_test'
                    evo['exploration'] = np.random.choice(['perturb', 'resample'])
                    self.learners.send(evo,dest=learner_source,tag=Tags.evolution_config)
                else:
                    print('Bottom Performer')
                    evo['evolution'] = True
                    evo['agent'] = agent_id
                    evo['ranking'] = ranking
                    if not self.elite == 0:
                        evo['evo_agent'] = self.rankings[np.random.choice(range(self.elite))]
                    else:
                        evo['evo_agent'] = self.rankings[0]
                    evo['evo_ranking'] = np.where(self.rankings == evo['evo_agent'])
                    evo['exploitation'] = 'truncation'
                    evo['exploration'] = np.random.choice(['perturb', 'resample'])
                    self.learners.send(evo,dest=learner_source,tag=Tags.evolution_config)
                print('MetaLearner Responded to evolution request with evolution config')

    def log(self, msg, to_print=False):
        text = "Meta\t\t{}".format(msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def close(self):
        self.menvs.Disconnect()

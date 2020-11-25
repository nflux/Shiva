import sys, time,traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import numpy as np
import pandas as pd
pd.set_option('max_columns', 4)

from shiva.core.admin import logger
from shiva.eval_envs.Evaluation import Evaluation
from shiva.helpers.misc import terminate_process
from shiva.helpers.utils.Tags import Tags
from shiva.core.admin import Admin
from shiva.envs.Environment import Environment


class MPIMultiEvaluationWrapper(Evaluation):
    """Manages one or more MPIEvaluations"""
    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.info = MPI.Status()
        self.sort = False # Flag for when to sort
        self.launch()

    def launch(self):
        """ Launches the MultiEvaluationWrapper and grabs Agents to be Evaluated

        Returns:
            None
        """
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEvaluationWrapper, self).__init__(self.configs)
        self._launch_evals()
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        self.log("Got config and evaluating Agent IDS: {}".format(self.agent_ids), verbose_level=1)

        '''Set functions and data structures'''
        if 'RoboCup' in self.configs['Environment']['type']:
            self.evaluations = pd.DataFrame(index=np.arange(0, self.num_agents), columns=self.eval_events+['total_score'])
            self.rankings = np.zeros(self.num_agents)
            self._sort_evals = getattr(self, '_sort_robocup')
            self._get_evaluations = getattr(self, '_get_robocup_evaluations')
        elif 'Gym' in self.configs['Environment']['type'] \
                or 'Unity' in self.configs['Environment']['type'] \
                or 'ParticleEnv' in self.configs['Environment']['type']:
            '''Here Evaluation metrics are hardcoded'''
            self.evaluations = pd.DataFrame(index=[id for id in self.agent_ids], columns=['Learner_ID', 'Role', 'Num_Evaluations', 'Average_Reward'])
            self.evaluations.index.name = "Agent_ID"
            self.evaluations['Learner_ID'] = self.evaluations.index.map(self.get_learner_id)
            self.evaluations['Role'] = self.evaluations.index.map(self.get_role)
            self.evaluations['Learner_ID'] = self.evaluations.index.map(self.get_learner_id)
            self.evaluations['Num_Evaluations'] = 0
            self.rankings = {role:[] for role in self.roles}
            self.current_matches = {eval_id:{} for eval_id in range(self.num_evals)}
            self._sort_evals = self._sort_roles
            self._get_evaluations = self._get_roles_evaluations
            self.initial_agent_selection = self._initial_role_selection
            self._get_initial_evaluations = self._get_initial_roles_evaluations
        else:
            self.evaluations = dict()
            self.rankings = np.zeros(self.num_agents)
            self._sort_evals = getattr(self, '_sort_simple')
            self._get_evaluations = getattr(self, '_get_simple_evaluations')

        self.log("Waiting start flag..", verbose_level=1)
        start_flag = self.meta.bcast(None, root=0)
        self.log("Start running..!!", verbose_level=1)
        self.run()

    def run(self):
        """ Sends Agents to get evaluated, waits for metrics, and ranks them based on the metrics.

        Returns:
            None
        """
        self.initial_agent_selection()
        self._get_initial_evaluations()

        while True:
            time.sleep(self.configs['Admin']['time_sleep']['EvalWrapper'])
            self._get_evaluations(sort=True)
            if self.sort:
                self._sort_evals()
        self.close()

    '''
        Roles Methods
    '''

    def _initial_role_selection(self):
        for i in range(self.num_evals):
            self._send_new_match(i)

    def _send_new_match(self, eval_id):
        new_match = self._get_new_match()
        self.current_matches[eval_id] = new_match # save reference to the match at that evaluation process
        self.evals.send(new_match, dest=eval_id, tag=Tags.new_agents)
        self.log("Sent new match {} to Eval_ID {}".format(new_match, eval_id), verbose_level=3)

    def _get_new_match(self) -> dict:
        """
            Creates a new match out of the population we have for each role
            With some probability @restrict_pairs_proba, we are strict to keep agent pairs together
                @restrict_pairs_proba = 1, to force pairs to be evaluated together

            * other approach, could be of creating a pool of possible matches (both with pairs and/or broken pairs) and grab a random one from the pool
        """
        match = {}
        # self.restrict_pairs_proba = 1
        # restrict_pairs = lambda: np.random.uniform() < self.restrict_pairs_proba
        match_is_full = lambda x: len(self.roles) == len(x.keys())

        for role in self.roles:
            if role in match:
                # role was filled by a companion of a random chosen agent
                continue

            '''Randomly choosing agents that haven't been evaluated as much'''
            # grab the ones with the less Num_Evaluations
            _min_num_evals_of_role = self.evaluations.query('Role == @role')['Num_Evaluations'].min()
            ids_with_min_num_evals = self.evaluations.query('Role == @role and Num_Evaluations == @_min_num_evals_of_role').index.tolist()
            # OPTIONAL: get IDs currently being evaluated and taken them out of @possible_ids to choose
            _take_out = [] # self.get_agents_currently_being_evaluated()

            possible_ids = list(set(ids_with_min_num_evals) - set(_take_out))
            if len(possible_ids) == 0:
                possible_ids = ids_with_min_num_evals
            # agent_id = np.random.choice(possible_ids)
            agent_id = possible_ids[0]

            match[role] = self.get_learner_spec(agent_id)

            '''Force pairs to play together'''
            pairs_roles = self.has_pair(agent_id, role)
            self.log("Found pair for {} to be {}".format(agent_id, pairs_roles), verbose_level=3)
            for role_of_pair in pairs_roles:
                match[role_of_pair] = match[role]

            if match_is_full(match):
                break
        return match

    def get_agents_currently_being_evaluated(self):
        """This function assumes that Agents from same Learner MUST be evaluated together

        Returns:
            List of Agent IDs
        """
        ret = []
        for eval_id, match in self.current_matches.items():
            for role, learner_spec in match.items():
                ret += learner_spec['role2ids'][role]
        return ret

    def has_pair(self, agent_id, role):
        """ Checks if agent id has another agent in the same learner so they can play together.

        Returns:
            A list of agent IDs.
        """
        learner_id = self.evaluations.at[agent_id, 'Learner_ID']
        return self.evaluations.query('Learner_ID == @learner_id')['Role'].tolist()

    def _get_initial_roles_evaluations(self):
        self.ready = False
        while not self.ready:
            self._get_evaluations(sort=False)

    def _get_roles_evaluations(self, sort):
        """Receive Evaluation results from one Evaluation processes"""
        while self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evals, status=self.info):
            eval_id = self.info.Get_source()
            eval_match = self.current_matches[eval_id]
            evals = self.evals.recv(None, source=eval_id, tag=Tags.evals)
            '''Update internal dataframe @self.evaluations'''
            for role, metrics in evals.items():
                # @metrics has only 'Average_Reward' for now
                '''Assuming a Learner has 1 Agent per Role'''
                agent_id = eval_match[role]['role2ids'][role][0]
                for metric_name, value in metrics.items():
                    self.evaluations.at[agent_id, metric_name] = value
                    self.evaluations.at[agent_id, 'Num_Evaluations'] += 1
            self.sort = sort
            self.ready = True
            self._send_new_match(eval_id)
            self.log("\nGot Evals {} for match {}".format(evals, eval_match), verbose_level=2)

    def _sort_roles(self):
        """Sort Evaluations and updates MetaLearner with current rankings"""
        '''Assuming 1 evaluation metric: Average Reward'''

        # skip if we don't have everyones rankings
        if self.evaluations['Average_Reward'].isna().sum() > 0:
            return

        self.evaluations.sort_values(by=['Role', 'Average_Reward'], ascending=False, inplace=True)
        for role in self.roles:
            self.rankings[role] = self.evaluations[self.evaluations['Role']==role].index.tolist()

        self.meta.send(self.rankings, dest=0, tag=Tags.rankings)
        self.log("Rankings\n{}".format(self.evaluations.sort_values(['Role', 'Average_Reward'], ascending=False)), verbose_level=1)

        self.sort = False

    def get_role(self, agent_id):
        for role, role_agent_ids in self.role2ids.items():
            if agent_id in role_agent_ids:
                return role

    def get_learner_id(self, agent_id):
        """ Returns the learner id that corresponds to passed agent id.

        Returns:
            Integer id representing the learner housing the agent.
        """
        for spec in self.learners_specs:
            if agent_id in spec['agent_ids']:
                return spec['id']

    def get_learner_spec(self, agent_id):
        """ Returns the specs for that learner containing the agent.

        Returns:
            Dictionary of specifications.
        """
        for spec in self.learners_specs:
            if agent_id in spec['agent_ids']:
                return spec

    '''
        Single Agent Methods
    '''

    def _get_initial_evaluations(self):
        while len(self.evaluations) < self.num_agents:
            self._get_evaluations(False)

    def _get_simple_evaluations(self, sort):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evals):
            agent_id = self.evals.recv(None, source=MPI.ANY_SOURCE, tag=Tags.evals, status=self.info)
            env_source = self.info.Get_source()
            evals = self.evals.recv(None, source=env_source, tag=Tags.evals)
            self.evaluations[agent_id] = evals.mean()
            self.sort = sort
            self.log('Multi Evaluation has received evaluations!', verbose_level=2)
            self.agent_selection(env_source)

    def _get_robocup_evaluations(self, sort):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag = Tags.evals):
            agent_id = self.evals.recv(None, source = MPI.ANY_SOURCE, tag=Tags.evals, status=self.info)
            env_source = self.info.Get_source()
            evals = self.evals.recv(None, source=env_source, tag=Tags.evals)
            #evals['agent_id'] = agent_id
            keys = evals[0].keys()
            averages = dict()
            for i in range(len(evals)):
                for key in keys:
                    if key in averages.keys():
                        averages[key] = averages[key] + ( (evals[i][key] - averages[key])/i)
                    else:
                        averages[key] = evals[i][key]
            self.evaluations.loc[agent_id,keys] = averages
            self.sort = sort
            self.log('Multi Evaluation has received evaluations', verbose_level=2)
            self.agent_selection(env_source)

    def _sort_simple(self):
        self.rankings = np.array(sorted(self.evaluations, key=self.evaluations.__getitem__, reverse=True))
        self.log('Rankings:\n{}'.format(self.rankings), verbose_level=1)
        self.meta.send(self.rankings, dest=0, tag=Tags.rankings)
        self.log('Sent rankings to Meta', verbose_level=2)
        self.sort = False

    def _sort_robocup(self):
        # self.log('Robocup Sort!')
        # self.evaluations['total_score'] = 0
        # self.evaluations= self.evaluations.rename_axis('agent_ids').reset_index()
        # for i,col in enumerate(self.eval_events):
        #     self.log('enumerating events')
        #     self.evaluations.sort_values(by=col,inplace=True)
        #     self.log('sorting')
        #     self.evaluations['total_score'] += np.array(self.evaluations.index) * self.eval_weights[i]
        #     self.log('setting total scores')
        # self.evaluations.sort_values(by='total_score',ascending=False,inplace=True)
        # self.log('Sorting Total Scores')
        self.evaluations.sort_values(by=self.eval_events, ascending=self.sort_ascending, inplace=True)
        # self.evaluations.sort_values(by=self.eval_events,inplace=True)
        self.rankings = np.array(self.evaluations.index)
        self.log('Rankings: {}'.format(self.rankings), verbose_level=2)
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations), verbose_level=1)
        self.evaluations.sort_index(inplace=True)
        self.meta.send(self.rankings, dest=0, tag=Tags.rankings)
        self.log('Sent rankings to Meta', verbose_level=2)
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations), verbose_level=2)
        self.sort = False


    def initial_agent_selection(self):
        """ Randomly selects the initial agents to be sent to which environment.

        Returns:
            None
        """
        for i in range(self.num_evals):
            self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))[0]
            print('Selected Evaluation Agents for Environment {}: {}'.format(i, self.agent_sel))
            self.evals.send(self.agent_sel,dest=i,tag=Tags.new_agents)

    def agent_selection(self, env_rank):
        """ Randomly selects the agents to be sent to which environment.

        Returns:
            None
        """
        self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))
        print('Selected Evaluation Agents for Environment {}: {}'.format(env_rank, self.agent_sel))
        self.evals.send(self.agent_sel, dest=env_rank,tag=Tags.new_agents)

    def _launch_evals(self):
        self.evals = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvaluation.py'], maxprocs=self.num_evals)
        self.evals.bcast(self.configs, root=MPI.ROOT)
        eval_specs = self.evals.gather(None, root=MPI.ROOT)
        assert len(eval_specs) == self.num_evals, "Not all Evaluations checked in.."
        self.eval_specs = eval_specs[0] # set self attr only 1 of them
        self.log("Got EvaluationSpecs {}".format(self.eval_specs), verbose_level=1)

    def _get_meval_specs(self):
        return {
            'type': 'MultiEval',
            'id': self.id,
            'eval_specs': self.eval_specs,
            'num_evals': self.num_evals
        }

    def close(self):
        """ Closes the connection with the MetaLearner.
        Returns:
            None
        """
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False, verbose_level=-1):
        """If verbose_level is not given, by default will log
        Args:
            msg: Message to be outputted
            to_print: Whether you want it as print or debug statement
            verbose_level: Level required for the message to be printed.
        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['EvalWrapper']:
            text = "{}\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<MultiEval(id={})>".format(self.id)

    def show_comms(self):
        """ Shows who this MultiEvalWrapper is connected to.
        Returns:
            None
        """
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))

if __name__ == "__main__":
    try:
        MPIMultiEvaluationWrapper()
    except Exception as e:
        msg = "<MultiEval(id={})> error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
    finally:
        terminate_process()

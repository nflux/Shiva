import sys, time,traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import numpy as np
import pandas as pd


from shiva.core.admin import logger
from shiva.eval_envs.Evaluation import Evaluation
from shiva.helpers.misc import terminate_process
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin
from shiva.envs.Environment import Environment

class MPIMultiEvaluationWrapper(Evaluation):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.info = MPI.Status()
        self.sort = False # Flag for when to sort
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEvaluationWrapper, self).__init__(self.configs)
        self._launch_evals()
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        self.log("Got config and evaluating Agent IDS: {}".format(self.agent_ids))

        if 'RoboCup' in self.configs['Environment']['type']:
            self.evaluations = pd.DataFrame(index=np.arange(0, self.num_agents), columns=self.eval_events+['total_score'])
            self.rankings = np.zeros(self.num_agents)
            self._sort_evals = getattr(self, '_sort_robocup')
            self._get_evaluations = getattr(self, '_get_robocup_evaluations')
        elif 'Unity' in self.configs['Environment']['type'] or 'ParticleEnv' in self.configs['Environment']['type']:
            '''Here Evaluation metrics are hardcoded'''
            self.evaluations = pd.DataFrame(index=[id for id in self.agent_ids], columns=['Role', 'Average_Reward'])
            self.evaluations['Role'] = self.evaluations.index.map(self.get_role)
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

        self.run()

    def run(self):
        self.log('MultiEvalWrapper start Running')
        self.initial_agent_selection()
        self._get_initial_evaluations()

        while True:
            time.sleep(0.1)
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

    def _get_new_match(self) -> dict:
        '''
            Creates a new match out of the population we have for each role
            With some probability @restrict_pairs_proba, we are strict to keep agent pairs together
                @restrict_pairs_proba = 1, to force pairs to be evaluated together

            * other approach, could be of creating a pool of possible matches (both with pairs and/or broken pairs) and grab a random one from the pool
        '''
        match = {}
        self.restrict_pairs_proba = 1
        restrict_pairs = lambda: np.random.uniform() < self.restrict_pairs_proba
        is_full = lambda x: len(self.roles) == len(x.keys())
        for role in self.roles:
            agent_id = np.random.choice(self.role2ids[role])
            match[role] = agent_id
            if restrict_pairs():  # probability function
                pair_bool, pairs = self.has_pair(agent_id, role)
                if pair_bool:
                    for pair_role, pair_id in pairs.items():
                        '''Assuming that Learners have only 1 Agent per Role'''
                        match[pair_role] = pair_id[0]
                if is_full(match):
                    break
        return match

    def has_pair(self, agent_id, role):
        has = False
        for l_spec in self.learner_specs:
            if agent_id in l_spec['agent_ids'] and len(l_spec['agent_ids']) > 1:
                for l_roles in l_spec['roles']:
                    if role != l_roles:
                        has = True
                if has:
                    return True, l_spec['role2ids']
        return False, []

    def _get_initial_roles_evaluations(self):
        self.ready = False
        while not self.ready:
            self._get_evaluations(sort=False)

    def _get_roles_evaluations(self, sort):
        '''Receive Evaluation results from one Evaluation processes'''
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evals, status=self.info):
            eval_id = self.info.Get_source()
            eval_match = self.current_matches[eval_id]
            evals = self.evals.recv(None, source=eval_id, tag=Tags.evals)
            '''Update internal dataframe @self.evaluations'''
            for role, metrics in evals.items():
                # @metrics has only 'Average_Reward' for now
                agent_id = eval_match[role]
                for metric_name, value in metrics.items():
                    self.evaluations.at[agent_id, metric_name] = value
            self.sort = sort
            self.ready = True
            self._send_new_match(eval_id)
            self.log("\nGot Tags.evals {} for match {}".format(evals, eval_match))

    def _sort_roles(self):
        '''Sort Evaluations and updates MetaLearner'''
        '''Assuming 1 evaluation metric: Average Reward'''
        self.evaluations.sort_values(by=['Role', 'Average_Reward'], ascending=False, inplace=True)

        for role in self.roles:
            self.rankings[role] = self.evaluations[self.evaluations['Role']==role].index.tolist()
        self.meta.send(self.rankings, dest=0, tag=Tags.rankings)
        self.log("\n{}\n{}".format(self.evaluations, self.rankings))
        self.sort = False

    def get_role(self, agent_id):
        for role, role_agent_ids in self.role2ids.items():
            if agent_id in role_agent_ids:
                return role

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
            self.log('Multi Evaluation has received evaluations!')
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
            self.log('Multi Evaluation has received evaluations')
            self.agent_selection(env_source)

    def _sort_simple(self):
        self.rankings = np.array(sorted(self.evaluations, key=self.evaluations.__getitem__, reverse=True))
        self.log('Rankings: {}'.format(self.rankings))
        self.meta.send(self.rankings,dest= 0,tag=Tags.rankings)
        self.log('Sent rankings to Meta')
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
        self.log('Rankings: {}'.format(self.rankings))
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations))
        self.evaluations.sort_index(inplace=True)
        self.meta.send(self.rankings,dest= 0,tag=Tags.rankings)
        self.log('Sent rankings to Meta')
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations))
        self.sort = False


    def initial_agent_selection(self):
        for i in range(self.num_evals):
            self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))[0]
            print('Selected Evaluation Agents for Environment {}: {}'.format(i, self.agent_sel))
            self.evals.send(self.agent_sel,dest=i,tag=Tags.new_agents)

    def agent_selection(self, env_rank):
        self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))
        print('Selected Evaluation Agents for Environment {}: {}'.format(env_rank, self.agent_sel))
        self.evals.send(self.agent_sel, dest=env_rank,tag=Tags.new_agents)

    def _launch_evals(self):
        self.evals = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvaluation.py'], maxprocs=self.num_evals)
        self.evals.bcast(self.configs, root=MPI.ROOT)
        eval_specs = self.evals.gather(None, root=MPI.ROOT)
        assert len(eval_specs) == self.num_evals, "Not all Evaluations checked in.."
        self.eval_specs = eval_specs[0] # set self attr only 1 of them
        self.log("Got EvaluationSpecs {}".format(self.eval_specs))

    def _get_meval_specs(self):
        return {
            'type': 'MultiEval',
            'id': self.id,
            'eval_specs': self.eval_specs,
            'num_evals': self.num_evals
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = "{}\t{}".format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<MultiEval(id={})>".format(self.id)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))

if __name__ == "__main__":
    try:
        MPIMultiEvaluationWrapper()
    except Exception as e:
        print("Eval Wrapper error:", traceback.format_exc())
    finally:
        terminate_process()

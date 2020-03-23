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
        self.sort = False #Flag for when to sort
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiEvaluationWrapper, self).__init__(self.configs)
        self._launch_evals()
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        self.info = MPI.Status()
        self.agent_ids = self.meta.bcast(None,root=0)
        self.log('Agent IDs being evaluated: {}'.format(self.agent_ids))
        if 'RoboCup' in self.configs['Environment']['type']:
            self.evaluations = pd.DataFrame(index=np.arange(0, self.num_agents), columns=self.eval_events+['total_score'])
            self.rankings = np.zeros(self.num_agents)
            self._sort_evals = getattr(self, '_sort_robocup')
            self._get_evaluations = getattr(self, '_get_robocup_evaluations')
        else:
            self.evaluations = dict()
            self.rankings = np.zeros(self.num_agents)
            self._sort_evals = getattr(self, '_sort_simple')
            self._get_evaluations = getattr(self, '_get_simple_evaluations')
        self.initial_agent_selection()
        self.run()

    def run(self):
        self.log('MultiEvalWrapper is running')
        self._get_initial_evaluations()
        while True:
            time.sleep(0.1)
            self._get_evaluations(True)
            if self.sort:
                self._sort_evals()
        self.close()

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

    def _get_robocup_evaluations(self,sort):
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
        #self.log('Robocup Sort!')
        #self.evaluations['total_score'] = 0
        #self.evaluations= self.evaluations.rename_axis('agent_ids').reset_index()
        #for i,col in enumerate(self.eval_events):
            #self.log('enumerating events')
            #self.evaluations.sort_values(by=col,inplace=True)
            #self.log('sorting')
            #self.evaluations['total_score'] += np.array(self.evaluations.index) * self.eval_weights[i]
            #self.log('setting total scores')
        #self.evaluations.sort_values(by='total_score',ascending=False,inplace=True)
        #self.log('Sorting Total Scores')
        self.evaluations.sort_values(by=self.eval_events, ascending=self.sort_ascending, inplace=True)
        #self.evaluations.sort_values(by=self.eval_events,inplace=True)
        self.rankings = np.array(self.evaluations.index)
        self.log('Rankings: {}'.format(self.rankings))
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations))
        self.evaluations.sort_index(inplace=True)
        self.meta.send(self.rankings,dest= 0,tag=Tags.rankings)
        self.log('Sent rankings to Meta')
        self.log('Current Rankings DataFrame: {}'.format(self.evaluations))
        self.sort = False


    # def _get_evaluations(self,sort):
    #     if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag = Tags.evals):
    #         evals = self.evals.recv(None,source=MPI.ANY_SOURCE,tag=Tags.evals,status=self.info)
    #         eval_source = self.info.Get_source()
    #         self.log('Agent IDS: '.format(evals['agent_ids']))
    #         for i in range(len(evals['agent_ids'])):
    #             self.evaluations[evals['agent_ids'][i]] = evals['evals'][i].mean()
    #         self.sort=sort
    #         print('Multi Evaluation has received evaluations!')
    #         self.agent_selection(eval_source)

    def initial_agent_selection(self):
        '''Choose Agent to be Evaluated on each Environment'''
        for i in range(self.num_evals):
            self.agent_selection(i)
            # self.agent_sel = np.reshape(np.random.choice(self.agent_ids, size=self.agents_per_env, replace=False), (-1, self.agents_per_env))[0]
            # self.log("Selected Evaluation Agents for Environment {}: {}".format(i, self.agent_sel))
            # self.evals.send(self.agent_sel, dest=i, tag=Tags.new_agents)

    def agent_selection(self, env_id):
        '''Choose Agent to be Evaluated on each Environment'''
        self.agent_sel = np.reshape(np.random.choice(self.agent_ids, size=self.agents_per_env, replace=False), (-1, self.agents_per_env))
        self.log('Selected Evaluation Agents for Environment {}: {}'.format(env_id, self.agent_sel))
        self.evals.send(self.agent_sel, dest=env_id, tag=Tags.new_agents)

    def _launch_evals(self):
        self.evals = MPI.COMM_WORLD.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvaluation.py'], maxprocs=self.num_evals)
        self.evals.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        eval_spec = self.evals.gather(None, root=MPI.ROOT)  # Wait for Eval Specs ()
        assert len(eval_spec) == self.num_evals, "Not all Evaluations checked in.."
        self.eval_specs = eval_spec[0] # set self attr only 1 of them
        self.log("Got EvaluationSpecs {}".format(self.eval_spec))

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
        text = '<MultiEval(id={})>\t{}'.format(self.id, msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

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

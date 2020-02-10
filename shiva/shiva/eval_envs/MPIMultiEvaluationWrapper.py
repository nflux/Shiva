import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import numpy as np
import pandas as pd

from shiva.eval_envs.Evaluation import Evaluation
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin
from shiva.envs.Environment import Environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

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
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self.rankings = np.zeros(self.num_agents)
        self.evaluations = dict()
        self._launch_evals()
        self.meta.gather(self._get_meval_specs(), root=0) # checkin with Meta
        self.agent_ids = range(self.num_agents)
        self.agent_sel = np.reshape(np.random.choice(self.agent_ids,size = self.agents_per_env, replace=False),(-1,self.agents_per_env))
        print('Selected Evaluation Agents: ', self.agent_sel)
        print('\n\n\n\n\n')
        self.evals.scatter(self.agent_sel,root=MPI.ROOT)

        self.run()

    def run(self):
        self.log('MultiEvalWrapper is running')
        self._get_initial_evaluations()

        while True:
            self._get_evaluations()

            if self.sort:
                self.rankings = np.array(sorted(self.evaluations, key=self.evaluations.__getitem__))
                print('Rankings: ', self.rankings)
                print('Rankings type: ', type(self.rankings))
                self.meta.send(self.rankings,dest= 0,tag=Tags.rankings)
                print('Sent rankings to Meta')
                self.sort = False

        self.close()

    def _launch_evals(self):
        # Spawn Single Environments
        self.evals = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvaluation.py'], maxprocs=self.num_evals)
        self.evals.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        #self.log('Eval configs sent')
        eval_spec = self.evals.gather(None, root=MPI.ROOT)  # Wait for Eval Specs ()
        #self.log('Eval specs received')
        assert len(eval_spec) == self.num_evals, "Not all Evaluations checked in.."
        self.eval_specs = eval_spec[0] # set self attr only 1 of them


    def _get_meval_specs(self):
        return {
            'type': 'MultiEval',
            'id': self.id,
            'eval_specs': self.eval_specs,
            'num_evals': self.num_evals
        }

    def _get_evaluations(self):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.agent_id):
            info = MPI.Status()
            agent_id = self.evals.recv(None, source=MPI.ANY_SOURCE, tag=Tags.agent_id, status=info)
            env_source = info.Get_source()
            eval = self.evals.recv(None, source=env_source, tag=Tags.evals)
            self.evaluations[agent_id] = eval.mean()
            self.sort = True
            print('Multi Evaluation has received evaluations!')

    def _get_initial_evaluations(self):
        while len(self.evaluations ) < self.configs['MetaLearner']['num_learners']:
            self._get_evaluations()
            #print('Multi Evaluations: ',self.evaluations)


    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'MultiEval {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))



if __name__ == "__main__":
    MPIMultiEvaluationWrapper()

from mpi4py import MPI
import sys, time, traceback, os
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
from shiva.core.admin import logger
from shiva.helpers.misc import terminate_process
import shiva.helpers.file_handler as fh
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin



class IOHandler(object):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.configs = self.meta.recv(None,source=0,tag=Tags.configs)
        self.learners_port = MPI.Open_port(MPI.INFO_NULL)
        self.menvs_port = MPI.Open_port(MPI.INFO_NULL)
        if self.configs['MetaLearner']['pbt']:
            self.evals_port = MPI.Open_port(MPI.INFO_NULL)
        self.specs = self._get_io_specs()
        self.log('Opened Ports: {}'.format(self.specs))
        self.meta.send(self.specs,dest=0,tag=Tags.io_config)
        self.log('Sent Meta my specs')
        self._connect_ports()
        self.log('Ports have been connected')
        self.info = MPI.Status()
        self.evo_evals = dict()
        self.run()

    def run(self):

        while True:
            time.sleep(0.001)
            self.service_learner_requests()
            self.service_menv_requests()
            if self.configs['MetaLearner']['pbt']:
                self.service_eval_requests()





    def _get_io_specs(self):
        if self.configs['MetaLearner']['pbt']:
            return {
            'learners_port': self.learners_port,
            'menvs_port': self.menvs_port,
            'evals_port': self.evals_port
            }
        else:
            return {
            'learners_port': self.learners_port,
            'menvs_port': self.menvs_port,
            }


    def _connect_ports(self):
        self.menvs = MPI.COMM_WORLD.Accept(self.menvs_port)
        self.log('MEnv Port Connected')
        self.learners = MPI.COMM_WORLD.Accept(self.learners_port)
        self.log('Learner Port Connected')
        if self.configs['MetaLearner']['pbt']:
            self.evals = MPI.COMM_WORLD.Accept(self.evals_port)
            self.log('Evals Port Connected')


    def service_learner_requests(self):
        if self.learners.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_learner_request):
            _ = self.learners.recv(None,source=MPI.ANY_SOURCE,tag=Tags.io_learner_request,status=self.info)
            source = self.info.Get_source()
            self.learners.send(True, dest=source,tag=Tags.io_learner_request)
            _ = self.learners.recv(None, source=source,tag=Tags.io_learner_request)

    def service_menv_requests(self):
        if self.menvs.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_menv_request):
            if self.menvs.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_menv_request):
                self.menvs.recv(None  ,source=MPI.ANY_SOURCE,tag=Tags.io_menv_request,status=self.info)
                source = self.info.Get_source()
                self.menvs.send(True, dest=source,tag=Tags.io_menv_request)
                _ = self.menvs.recv(None, source=source,tag=Tags.io_menv_request)


    def service_eval_requests(self):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_eval_request):
            _ = self.evals.recv(None,source=MPI.ANY_SOURCE,tag=Tags.io_eval_request,status=self.info)
            source = self.info.Get_source()
            self.evals.send(True, dest=source,tag=Tags.io_eval_request)
            _ = self.evals.recv(None, source=source,tag=Tags.io_eval_request)


    def log(self, msg, to_print=False):
        text = 'IOHandler: {}'.format(msg)
        logger.info(text, True)



if __name__ == "__main__":
    try:
        IOHandler()
    except Exception as e:
        print("Eval Wrapper error:", traceback.format_exc())
    finally:
        terminate_process()

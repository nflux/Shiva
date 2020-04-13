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
        self.configs = self.meta.recv(None, source=0, tag=Tags.configs)
        self.learners_port = MPI.Open_port(MPI.INFO_NULL)
        self.menvs_port = MPI.Open_port(MPI.INFO_NULL)
        self.evals_port = MPI.Open_port(MPI.INFO_NULL)

        self.specs = self._get_io_specs()
        self.log('Opened Ports: {}'.format(self.specs))
        self.meta.send(self.specs, dest=0, tag=Tags.io_config)

        self._connect_ports()
        self.info = MPI.Status()
        self.run()

    def run(self):
        self.dirs_in_use = []
        while True:
            time.sleep(0.001)
            self.service_learner_requests()
            self.service_menv_requests()
            if self.configs['MetaLearner']['pbt']:
                self.service_eval_requests()

    def _connect_ports(self):
        self.menvs = MPI.COMM_WORLD.Accept(self.menvs_port)
        self.log('MultiEnv Port Connected', verbose_level=1)
        self.learners = MPI.COMM_WORLD.Accept(self.learners_port)
        self.log('Learner Port Connected', verbose_level=1)
        if self.configs['MetaLearner']['pbt']:
            self.evals = MPI.COMM_WORLD.Accept(self.evals_port)
            self.log('Evals Port Connected', verbose_level=1)

    def service_learner_requests(self):
        tag = Tags.io_learner_request
        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=tag, status=self.info):
            source = self.info.Get_source()
            self.log("Learner {} has IO access".format(source), verbose_level=2)
            _ = self.learners.recv(None, source=source, tag=tag)
            self.learners.send(True, dest=source, tag=tag)
            _ = self.learners.recv(None, source=source, tag=tag)

    def service_menv_requests(self):
        tag = Tags.io_menv_request
        if self.menvs.Iprobe(source=MPI.ANY_SOURCE, tag=tag, status=self.info):
            source = self.info.Get_source()
            self.log("MultiEnv {} has IO access".format(source), verbose_level=2)
            self.menvs.recv(None, source=source, tag=tag)
            self.menvs.send(True, dest=source,tag=tag)
            _ = self.menvs.recv(None, source=source, tag=tag)

    def service_eval_requests(self):
        tag = Tags.io_eval_request
        if self.evals.Iprobe(source=MPI.ANY_SOURCE, tag=tag, status=self.info):
            source = self.info.Get_source()
            self.log("Eval {} has IO access".format(source), verbose_level=2)
            _ = self.evals.recv(None, source=source, tag=tag)
            self.evals.send(True, dest=source, tag=tag)
            _ = self.evals.recv(None, source=source, tag=tag)

    def _service_request(self, comm, source, tag):
        req = comm.recv(None, source=source, tag=tag)
        if req['dir'] in self.dirs_in_use:

        else:
            comm.send(True, dest=source, tag=tag)
            _ = comm.

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

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['IOHandler']:
            text = '{}\t\t{}'.format(str(self), msg)
            logger.info(text, to_print=to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<IO>"

if __name__ == "__main__":
    try:
        IOHandler()
    except Exception as e:
        print("IOHandler error: ", traceback.format_exc())
    finally:
        terminate_process()
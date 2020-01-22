#!/usr/bin/env python
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI

from shiva.envs.Environment import Environment
from shiva.helpers.config_handler import load_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

class MPIEnv(Environment):

    def __init__(self):
        self.menv = MPI.Comm.Get_parent()
        self.id = self.menv.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from MultiEnv
        self.configs = self.menv.bcast(None, root=0)
        super(MPIEnv, self).__init__(self.configs)
        self.debug("Received config with {} keys".format(len(self.configs.keys())))
        self._launch_env()
        self._connect_learners()
        # Check-in with MultiEnv that successfully connected with Learner
        self.menv.gather(self._get_env_specs(), root=0)
        self._create_buffer()
        # Wait for flag to start running
        self.debug("Waiting MultiEnv flag to start")
        start_flag = self.menv.bcast(None, root=0)
        self.debug("Start collecting..")
        self.collect = True
        self.run()

    def run(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.done = []

        self.env.reset()
        while self.collect:
            observations = list(self.env.get_observations())
            self.menv.gather(observations, root=0)
            actions = self.menv.scatter(None, root=0)
            next_observations, reward, done, _ = self.env.step(actions)
            # self.debug("{} {} {} {} {}".format(type(observations), type(actions), type(next_observations), type(reward), type(done)))

            self.observations.append(observations)
            self.actions.append(actions)
            self.next_observations.append(next_observations)
            self.rewards.append(reward)
            self.done.append(done)

            if self.env.is_done():
                traj = [self.observations, self.actions, self.rewards, self.next_observations, self.done]
                self.buffer.append(traj)
                self.debug(self.env.get_metrics(episodic=True))
                '''
                    Assuming 1 learner here
                        Spec should indicate what agent corresponds to that learner dest=ix
                '''
                for ix in range(self.num_learners):
                    self.learner.send(self._get_env_state(), dest=ix, tag=7)

                self.buffer = []
                self.env.reset()

        self.close()

    def _create_buffer(self):
        self.buffer = []

    def _launch_env(self):
        # initiate env from the config
        self.env = self.create_environment()
        # Check in and send single env specs with MultiEnv
        self.menv.gather(self._get_env_specs(), root=0)

    def _connect_learners(self):
        self.debug("Waiting Learners info")
        self.learners_specs = self.menv.bcast(None, root=0) # Wait until Learners info arrives from MultiEnv
        self.num_learners = len(self.learners_specs)
        # self.debug("Got Learners info and will connect with {} learners".format(self.num_learners))
        # Start communication with Learners
        self.learners_port = self.learners_specs[0]['port']
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        self.debug("Connected with {} learners on {}".format(self.num_learners, self.learners_port))

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def _get_env_state(self):
        return {
            'metrics': self.env.get_metrics(episodic=True),
            'buffer': self.buffer
        }

    def _get_env_specs(self):
        return {
            'type': 'Env',
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def debug(self, msg, to_print=False):
        text = "Env {}/{}\t\t{}".format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logging.debug(text)
        if to_print or self.configs['Admin']['debug']:
            print(text)

    def show_comms(self):
        self.debug("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.debug("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.debug("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    MPIEnv()
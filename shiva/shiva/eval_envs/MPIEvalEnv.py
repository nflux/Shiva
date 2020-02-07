#!/usr/bin/env python
import numpy as np
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.helpers.config_handler import load_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

class MPIEvalEnv(Environment):

    def __init__(self):
        #Comms to communicate with the MPI Evaluation Object
        self.eval = MPI.Comm.Get_parent()
        self.id = self.eval.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from MPI Evaluation Object
        self.configs = self.eval.bcast(None, root=0)
        super(MPIEvalEnv, self).__init__(self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        self._launch_env()
        # Check in and send single env specs with MPI Evaluation Object
        self.eval.gather(self._get_env_specs(), root=0)
        self.eval.gather(self._get_env_state(), root=0)
        #self._connect_learners()
        self._clear_buffers()
        # Wait for flag to start running
        self.debug("Waiting Eval flag to start")
        start_flag = self.eval.bcast(None, root=0)
        self.log("Start collecting..")

        self.run()

    def run(self):
        self.env.reset()


        while True:
            observations = list(self.env.get_observations())
            self.eval.gather(observations, root=0)
            actions = self.eval.scatter(None, root=0)
            next_observations, reward, done, _ = self.env.step(actions)

            # self.debug("{} {} {} {} {}".format(observations, actions, next_observations, reward, done))
            # self.debug("{} {} {} {} {}".format(type(observations), type(actions), type(next_observations), type(reward), type(done)))

            self.observations.append(observations)
            self.actions.append(actions)
            self.next_observations.append(next_observations)
            self.rewards.append(reward)
            self.done.append(done)

            if self.env.is_done():

                self.log(self.env.get_metrics(episodic=True)) # print metrics

                '''ASSUMING trajectory for 1 AGENT on both approaches'''
                # self._send_trajectory_python_list()
                self._send_eval_numpy()

                self._clear_buffers()
                self.env.reset()


            '''Come back to this for emptying old evaluations when a new agent is Loaded

                if self.eval.bcast(None,root=0):
                self._clear_buffers()
                self.debug('Buffer has been cleared')
                self.env.reset()'''

        self.close()

    def _send_trajectory_python_list(self):
        '''Python List approach'''
        trajectory = [[self.observations, self.actions, self.rewards, self.next_observations, self.done]]
        '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner dest=ix'''
        for ix in range(self.num_learners):
            '''Python List Approach'''
            self.learner.send(self._get_env_state(trajectory), dest=ix, tag=Tags.trajectory)

    def _send_eval_numpy(self):
        '''Numpy approach'''

        self.episode_reward = np.array(self.rewards).sum()
        self.eval.send(self.env.steps_per_episode, dest=0, tag=Tags.trajectory_length)
        self.eval.send(self.episode_reward, dest=0, tag = Tags.trajectory_eval)

    def _clear_buffers(self):
        '''
            --NEED TO DO MORE RESEARCH ON THIS--
            Python List append is O(1)
            While Numpy concatenation needs to reallocate memory for the list, thus slower
            https://stackoverflow.com/questions/38470264/numpy-concatenate-is-slow-any-alternative-approach
        '''
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.rewards = []
        self.done = []

    def _launch_env(self):
        # initiate env from the config
        self.env = self.create_environment()

    def _connect_learners(self):
        self.log("Waiting Learners info")
        self.learners_specs = self.menv.bcast(None, root=0) # Wait until Learners info arrives from MultiEnv
        self.num_learners = len(self.learners_specs)
        # self.debug("Got Learners info and will connect with {} learners".format(self.num_learners))
        # Start communication with Learners
        self.learners_port = self.learners_specs[0]['port']
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        self.log("Connected with {} learners on {}".format(self.num_learners, self.learners_port))

    def create_environment(self):
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def _get_env_state(self, traj=[]):
        return {
            'metrics': self.env.get_metrics(episodic=True),
            'trajectory': traj
        }

    def _get_env_specs(self):
        return {
            'type': 'EvalEnv',
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            #'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'Learner {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    MPIEvalEnv()

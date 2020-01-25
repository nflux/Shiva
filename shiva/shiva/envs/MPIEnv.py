#!/usr/bin/env python
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI

from shiva.core.admin import logger
from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.helpers.config_handler import load_class

class MPIEnv(Environment):

    def __init__(self):
        self.menv = MPI.Comm.Get_parent()
        self.id = self.menv.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from MultiEnv
        self.configs = self.menv.bcast(None, root=0)
        super(MPIEnv, self).__init__(self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        self._launch_env()
        # Check in and send single env specs with MultiEnv
        self.menv.gather(self._get_env_specs(), root=0)
        self._connect_learners()
        # Check-in with MultiEnv that successfully connected with Learner
        self.menv.gather(self._get_env_state(), root=0)
        self._clear_buffers()
        # Wait for flag to start running
        self.log("Waiting MultiEnv flag to start")
        start_flag = self.menv.bcast(None, root=0)
        self.log("Start collecting..")

        self.run()

    def run(self):
        self.env.reset()

        while True:
            '''We could optimize this gather/scatter ops using numpys'''
            observations = list(self.env.get_observations())
            # self.log("Obs {}".format(observations))
            self.menv.gather(observations, root=0)
            actions = self.menv.scatter(None, root=0)
            # self.log("Act {}".format(actions))
            next_observations, reward, done, _ = self.env.step(actions)

            # self.log("{} {} {} {} {}".format(observations, actions, next_observations, reward, done))
            # self.log("{} {} {} {} {}".format(type(observations), type(actions), type(next_observations), type(reward), type(done)))

            self.observations.append(observations)
            self.actions.append(actions)
            self.next_observations.append(next_observations)
            self.rewards.append(reward)
            self.done.append(done)

            if self.env.is_done():

                self.log(self.env.get_metrics(episodic=True)) # print metrics

                '''ASSUMING trajectory for 1 AGENT on both approaches'''
                # self._send_trajectory_python_list()
                self._send_trajectory_numpy()

                self._clear_buffers()
                self.env.reset()

        self.close()

    def _send_trajectory_python_list(self):
        '''Python List approach'''
        trajectory = [[self.observations, self.actions, self.rewards, self.next_observations, self.done]]
        '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner dest=ix'''
        for ix in range(self.num_learners):
            '''Python List Approach'''
            self.learner.send(self._get_env_state(trajectory), dest=ix, tag=Tags.trajectory)

    def _send_trajectory_numpy(self):
        '''Numpy approach'''
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.next_observations = np.array(self.next_observations)
        self.rewards = np.array(self.rewards)
        self.done = np.array(self.done)
        '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner, use dest=ix'''
        for ix in range(self.num_learners):
            self.learner.send(self.env.steps_per_episode, dest=ix, tag=Tags.trajectory_length)
            self.learner.Send([self.observations, MPI.FLOAT], dest=ix, tag=Tags.trajectory_observations)
            self.learner.Send([self.actions, MPI.FLOAT], dest=ix, tag=Tags.trajectory_actions)
            self.learner.Send([self.rewards, MPI.FLOAT], dest=ix, tag=Tags.trajectory_rewards)
            self.learner.Send([self.next_observations, MPI.FLOAT], dest=ix, tag=Tags.trajectory_next_observations)
            self.learner.Send([self.done, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)

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
        # self.log("Got Learners info and will connect with {} learners".format(self.num_learners))
        # Start communication with Learners
        self.learners_port = self.learners_specs[0]['port']
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        self.log("Connected with {} learners on {}".format(self.num_learners, self.learners_port))

    def create_environment(self):
        self.configs['Environment']['port'] = 5005 + (self.id * 10)
        self.configs['Environment']['worker_id'] = self.id * 11
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def _get_env_state(self, traj=[]):
        return {
            'metrics': self.env.get_metrics(episodic=True),
            'trajectory': traj
        }

    def _get_env_specs(self):
        return {
            'type': 'Env',
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents if hasattr(self.env, 'num_agents') else self.env.num_left+self.env.num_right,
            'num_instances_per_env': self.env.num_instances if hasattr(self.env, 'num_instances') else 1, # Unity case!!!!
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = "Env {}/{}\t\t{}".format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    # def print(self, msg):
    #     text = "Env {}/{}\t\t{}".format(self.id, MPI.COMM_WORLD.Get_size(), msg)
    #     print(text)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    MPIEnv()
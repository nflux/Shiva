#!/usr/bin/env python
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import torch

from shiva.core.admin import logger
from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.buffers.TensorBuffer import MultiAgentTensorBuffer
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
        self.create_buffers()
        # Wait for flag to start running
        self.log("Waiting MultiEnv flag to start")
        start_flag = self.menv.bcast(None, root=0)
        self.log("Start collecting..")

        self.run()

    def run(self):
        self.env.reset()
        while True:
            self._step_numpy()
            self._append_step()

            if self.env.is_done():
                self.print(self.env.get_metrics(episodic=True)) # print metrics
                self._send_trajectory_numpy()
                self.env.reset()
        self.close()

    def _step_numpy(self):
        self.observations = self.env.get_observations()
        '''
            Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
        '''
        # self.log("Obs Shape {}".format(self.observations.shape))
        send_obs_buffer = np.array(self.observations)
        self.menv.gather(send_obs_buffer, root=0)
        self.actions = self.menv.scatter(None, root=0)
        # self.log("Act {}".format(self.actions))
        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)

        # self.log("Step shape\tObs {}\tAcs {}\tNextObs {}\tReward {}\tDones{}".format(np.array(self.observations).shape, np.array(self.actions).shape, np.array(self.next_observations).shape, np.array(self.reward).shape, np.array(self.done).shape))
        # self.log("Actual types: {} {} {} {} {}".format(type(self.observations), type(self.actions), type(self.next_observations), type(self.reward), type(self.done)))

    def _append_step(self):
        for ix, buffer in enumerate(self.trajectory_buffers):
            exp = list(map(torch.clone, (torch.tensor([self.observations[ix]]),
                                         torch.tensor([self.actions[ix]]),
                                         torch.tensor([self.rewards[ix]]).unsqueeze(dim=-1),
                                         torch.tensor([self.next_observations[ix]]),
                                         torch.tensor([self.dones[ix]], dtype=torch.bool).unsqueeze(dim=-1)
                                         )))
            buffer.push(exp)

    def _unity_reshape(self, arr):
        '''Unity reshape of the data - concat all agents trajectories'''
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, 1, dim))

    def _send_trajectory_numpy(self):
        if 'Unity' in self.type:
            for ix in range(self.num_learners):
                # obs, acs, rew, next_obs, done = self.trajectory_buffers[ix].all_numpy()
                # self.observations_buffer = self._my_reshape(obs)
                # self.actions_buffer = self._my_reshape(acs)
                # self.rewards_buffer = self._my_reshape(rew)
                # self.next_observations_buffer = self._my_reshape(next_obs)
                # self.done_buffer = self._my_reshape(done)

                '''Assuming 1 Agent per Learner, no support for MADDPG here'''
                obs_buffer, acs_buffer, rew_buffer, next_obs_buffer, done_buffer = map(self._unity_reshape, self.trajectory_buffers[ix].all_numpy())

                self.log("Sending to Learner {} Obs shape {} Acs shape {} Rew shape {} NextObs shape {} Dones shape {}".format(ix, obs_buffer.shape, acs_buffer.shape, rew_buffer.shape, next_obs_buffer.shape, done_buffer.shape))

                self.learner.send(self.env.steps_per_episode, dest=ix, tag=Tags.trajectory_length)
                self.learner.Send([obs_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_observations)
                self.learner.Send([acs_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_actions)
                self.learner.Send([rew_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_rewards)
                self.learner.Send([next_obs_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_next_observations)
                self.learner.Send([done_buffer, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)
            self.reset_buffers()

        else:
            assert "Not Tested"
            self.observations_buffer = np.array(self.observations_buffer)
            self.actions_buffer = np.array(self.actions_buffer)
            self.next_observations_buffer = np.array(self.next_observations_buffer)
            self.rewards_buffer = np.array(self.rewards_buffer)
            self.done_buffer = np.array(self.done_buffer)
            '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner, use dest=ix'''
            self.log(
                "Trajectory shape: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(
                    self.observations_buffer.shape, self.actions_buffer.shape,
                    self.rewards_buffer.shape,
                    self.next_observations_buffer.shape,
                    self.done_buffer.shape))
            self.learner.send(self.env.steps_per_episode, dest=ix, tag=Tags.trajectory_length)
            self.learner.Send([self.observations_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_observations)
            self.learner.Send([self.actions_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_actions)
            self.learner.Send([self.rewards_buffer, MPI.FLOAT], dest=ix, tag=Tags.trajectory_rewards)
            self.learner.Send([self.next_observations_buffer, MPI.FLOAT], dest=ix,
                              tag=Tags.trajectory_next_observations)
            self.learner.Send([self.done_buffer, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)

    def create_buffers(self):
        if 'Unity' in self.type:
            '''
                Need a buffer for each agent group (brain)
                Unity is a bit different due to the multi-instance per simulation!
                Each Agent Group may have many agent IDs
            '''
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length, self.env.num_instances_per_group[group], self.env.observation_space[group], self.env.action_space[group]['acs_space']) for i, group in enumerate(self.env.agent_groups) ]
        else:
            assert "MPIEnv.py Buffers NotImplemented"

    def reset_buffers(self):
        if 'Unity' in self.type:
            for buffer in self.trajectory_buffers:
                buffer.reset()
        else:
            assert "MPIEnv.py Buffers NotImplemented"

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
        self.configs['Environment']['port'] += (self.id * 10)
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
            'num_agents': self.env.num_agents,
            'num_instances_per_env': self.env.num_instances_per_env if hasattr(self.env, 'num_instances_per_env') else 1, # Unity case
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = "Env {}/{}\t\t{}".format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def print(self, msg):
        text = "Env {}/{}\t\t{}".format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        print(text)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


    # def _step_python_list(self):
    #     self.observations = list(self.env.get_observations().values())
    #     # self.log("Obs {}".format(observations))
    #     self.menv.gather(self.observations, root=0)
    #     self.actions = self.menv.scatter(None, root=0)
    #     self.log("Act {}".format(self.actions))
    #     self.next_observations, self.reward, self.done, _ = self.env.step(self.actions)

    # def _send_trajectory_python_list(self):
    #     '''Python List approach'''
    #     trajectory = [[self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer]]
    #     '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner dest=ix'''
    #     for ix in range(self.num_learners):
    #         '''Python List Approach'''
    #         self.learner.send(self._get_env_state(trajectory), dest=ix, tag=Tags.trajectory)

if __name__ == "__main__":
    MPIEnv()
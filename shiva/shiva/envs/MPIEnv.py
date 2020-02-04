#!/usr/bin/env python
import numpy as np
import sys, traceback
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
                # self.print(self.env.get_metrics(episodic=True)) # print metrics
                self.log("Is done now")
                self._send_trajectory_numpy()
                self.env.reset()
        self.close()

    def _step_numpy(self):
        self.log("Not getting obs")
        self.observations = self.env.get_observations()
        '''
            Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
        '''
        # self.log("Obs Shape {}".format(self.observations.shape))
        send_obs_buffer = np.array(self.observations, dtype=np.float64)
        self.log("Obs Shape Send {}".format(send_obs_buffer.shape))
        self.menv.Gather([send_obs_buffer, MPI.DOUBLE], None, root=0)

        recv_action = np.zeros((self.env.num_agents, sum(self.env.action_space.values())), dtype=np.float64)
        self.log("The recv action {}".format(recv_action.shape))
        self.menv.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
        self.actions = recv_action
        # self.log("After getting actions")
        # self.log("Act {}".format(self.actions))
        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)

        self.log("Step shape\tObs {}\tAcs {}\tNextObs {}\tReward {}\tDones{}".format(np.array(self.observations).shape, np.array(self.actions).shape, np.array(self.next_observations).shape, np.array(self.rewards).shape, np.array(self.dones).shape))
        # self.log("Actual types: {} {} {} {} {}".format(type(self.observations), type(self.actions), type(self.next_observations), type(self.reward), type(self.done)))

    def _append_step(self):
        if 'Unity' in self.type:
            for ix, buffer in enumerate(self.trajectory_buffers):
                exp = list(map(torch.clone, (torch.tensor([self.observations[ix]]),
                                            torch.tensor([self.actions[ix]]),
                                            torch.tensor([self.rewards[ix]]).unsqueeze(dim=-1),
                                            torch.tensor([self.next_observations[ix]]),
                                            torch.tensor([self.dones[ix]], dtype=torch.bool).unsqueeze(dim=-1)
                                            )))
                buffer.push(exp)
        else:
            # self.log("Step shape\tObs {}\tAcs {}\tNextObs {}\tReward {}\tDones{}".format(np.array(self.observations).shape, np.array(self.actions).shape, np.array(self.next_observations).shape, np.array(self.rewards).shape, np.array(self.dones).shape))
            exp = list(map(torch.clone, (torch.tensor([self.observations]),
                                            torch.tensor([self.actions]),
                                            torch.tensor([self.rewards]).unsqueeze(dim=-1),
                                            torch.tensor([self.next_observations]),
                                            torch.tensor([self.dones], dtype=torch.bool).unsqueeze(dim=-1)
                                            )))

            self.log("Pushing to env buffer")
            self.trajectory_buffer.push(exp)

    def _unity_reshape(self, arr):
        '''Unity reshape of the data - concat all agents trajectories'''
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, 1, dim))
    
    def _reshape(self, arr):
        arr = np.ascontiguousarray(arr)
        traj, dim = arr.shape
        return np.reshape(arr, (traj, 1, dim))

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
                self.learner.Send([obs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_observations)
                self.learner.Send([acs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_actions)
                self.learner.Send([rew_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_rewards)
                self.learner.Send([next_obs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_next_observations)
                self.learner.Send([done_buffer, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)

            self.reset_buffers()
        else:
            '''Assuming 1 Agent per Learner, no support for MADDPG here'''
            for ix in range(self.num_learners):
                obs_buffer, acs_buffer, rew_buffer, next_obs_buffer, done_buffer = map(self._reshape, self.trajectory_buffer.agent_numpy(ix))

                self.log(
                "Trajectory shape: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(
                    obs_buffer.shape, acs_buffer.shape,
                    rew_buffer.shape,
                    next_obs_buffer.shape,
                    done_buffer.shape))
            
                self.learner.send(self.env.steps_per_episode, dest=ix, tag=Tags.trajectory_length)
                self.learner.Send([obs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_observations)
                self.learner.Send([acs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_actions)
                self.learner.Send([rew_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_rewards)
                self.learner.Send([next_obs_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_next_observations)
                self.learner.Send([done_buffer, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)
            
            self.reset_buffer()

    def create_buffers(self):
        if 'Unity' in self.type:
            '''
                Need a buffer for each agent group (brain)
                Unity is a bit different due to the multi-instance per simulation!
                Each Agent Group may have many agent IDs
            '''
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length, self.env.num_instances_per_group[group], self.env.observation_space[group], self.env.action_space[group]['acs_space']) for i, group in enumerate(self.env.agent_groups) ]
        else:
            self.trajectory_buffer = MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length, self.env.num_agents, self.env.get_observation_space(), self.env.get_action_space())

    def reset_buffers(self):
        for buffer in self.trajectory_buffers:
            buffer.reset()
    
    def reset_buffer(self):
        self.trajectory_buffer.reset()

    # def run(self):
    #     self.env.reset()

    #     while True:
    #         '''We could optimize this gather/scatter ops using numpys'''
    #         observations = list(self.env.get_observations())
    #         # self.log("Obs {}".format(observations))
    #         self.menv.gather(observations, root=0)
    #         actions = self.menv.scatter(None, root=0)
    #         # self.log("Act {}".format(actions))
    #         next_observations, reward, done, _ = self.env.step(actions)

    #         # self.log("{} {} {} {} {}".format(observations, actions, next_observations, reward, done))
    #         # self.log("{} {} {} {} {}".format(type(observations), type(actions), type(next_observations), type(reward), type(done)))

    #         self.observations.append(observations)
    #         self.actions.append(actions)
    #         self.next_observations.append(next_observations)
    #         self.rewards.append(reward)
    #         self.done.append(done)

    #         if self.env.is_done():

    #             self.log(self.env.get_metrics(episodic=True)) # print metrics

    #             '''ASSUMING trajectory for 1 AGENT on both approaches'''
    #             # self._send_trajectory_python_list()
    #             self._send_trajectory_numpy()

    #             self._clear_buffers()
    #             self.env.reset()

    #     self.close()

    # def _send_trajectory_python_list(self):
    #     '''Python List approach'''
    #     trajectory = [[self.observations, self.actions, self.rewards, self.next_observations, self.done]]
    #     '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner dest=ix'''
    #     for ix in range(self.num_learners):
    #         '''Python List Approach'''
    #         self.learner.send(self._get_env_state(trajectory), dest=ix, tag=Tags.trajectory)

    # def _send_trajectory_numpy(self):
    #     '''Numpy approach'''
    #     self.observations = np.array(self.observations)
    #     self.actions = np.array(self.actions)
    #     self.next_observations = np.array(self.next_observations)
    #     self.rewards = np.array(self.rewards)
    #     self.done = np.array(self.done)
    #     '''Assuming 1 learner here --> Spec should indicate what agent corresponds to that learner, use dest=ix'''
    #     for ix in range(self.num_learners):
    #         self.learner.send(self.env.steps_per_episode, dest=ix, tag=Tags.trajectory_length)
    #         self.learner.Send([self.observations, MPI.FLOAT], dest=ix, tag=Tags.trajectory_observations)
    #         self.learner.Send([self.actions, MPI.FLOAT], dest=ix, tag=Tags.trajectory_actions)
    #         self.learner.Send([self.rewards, MPI.FLOAT], dest=ix, tag=Tags.trajectory_rewards)
    #         self.learner.Send([self.next_observations, MPI.FLOAT], dest=ix, tag=Tags.trajectory_next_observations)
    #         self.learner.Send([self.done, MPI.BOOL], dest=ix, tag=Tags.trajectory_dones)

    # def _clear_buffers(self):
    #     '''
    #         --NEED TO DO MORE RESEARCH ON THIS--
    #         Python List append is O(1)
    #         While Numpy concatenation needs to reallocate memory for the list, thus slower
    #         https://stackoverflow.com/questions/38470264/numpy-concatenate-is-slow-any-alternative-approach
    #     '''
    #     self.observations = []
    #     self.actions = []
    #     self.next_observations = []
    #     self.rewards = []
    #     self.done = []

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
            'type': self.env.type,
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.action_space,
            'num_agents': self.env.num_agents,
            'num_instances_per_env': self.env.num_instances_per_env if hasattr(self.env, 'num_instances_per_env') else 1, # Unity case
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        self.learner.Unpublish_name()
        self.learner.Close_port()
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()
        MPI.COMM_WORLD.Abort()

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
    env = MPIEnv()

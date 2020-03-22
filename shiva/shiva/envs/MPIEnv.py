#!/usr/bin/env python
import time, subprocess
import numpy as np
import sys, time, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import torch

from shiva.core.admin import logger
from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.buffers.TensorBuffer import MultiAgentTensorBuffer
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process

class MPIEnv(Environment):

    def __init__(self):
        self.menv = MPI.Comm.Get_parent()
        self.id = self.menv.Get_rank()
        self.launch()
        self.done_count = 0

    def launch(self):
        # Receive Config from MultiEnv
        self.configs = self.menv.bcast(None, root=0)
        super(MPIEnv, self).__init__(self.configs)
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self._launch_env()
        # Check in and send single env specs with MultiEnv
        self.menv.gather(self._get_env_specs(), root=0)
        self._connect_learners()
        self.create_buffers()
        # Wait for flag to start running
        #self.log("Waiting MultiEnv flag to start")
        start_flag = self.menv.bcast(None, root=0)
        #self.log("Start collecting..")

        self.run()

    def run(self):
        self.env.reset()

        '''Need to wait for any initialization overhead of the environment (Robocup)'''
        while not self.env.start_env():
            pass

        while True:
            self._step_python()
            # self._step_numpy()
            self._append_step()
            if self.env.is_done():
                self._send_trajectory_numpy()
                self.env.reset()
            '''Check if there a new Role->LearnerID mapping'''
            # if self.menv.Iprobe():
            #     '''Receive message'''
            #     '''Clear the buffer'''
            #     pass

    def _step_python(self):
        self.observations = self.env.get_observations()
        self.menv.gather(self.observations, root=0)
        self.actions = self.menv.scatter(None, root=0)
        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)
        # self.print("Obs {} Act {} Rew {}".format(self.observations, self.actions, self.rewards))

    def _step_numpy(self):
        self.observations = self.env.get_observations()
        '''Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
                                    --> # of instances of that Agent per EnvWrapper is usually 1, except Unity?
        '''
        self.observations = np.array(self.observations, dtype=np.float64)
        self.menv.Gather([self.observations, MPI.DOUBLE], None, root=0)

        if 'Gym' in self.type or 'RoboCup' in self.type:
            recv_action = np.zeros((self.env.num_agents, self.env.action_space['acs_space']), dtype=np.float64)
            self.menv.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
            self.actions = recv_action
        elif 'Unity':
            self.actions = self.menv.scatter(None, root=0)

        # self.log("Obs {} Act {}".format(self.observations, self.actions))
        # self.log("Act {}".format(self.actions))
        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)
       # self.log('The Dones look like this: {}'.format(self.dones))

        # self.log("Step shape\tObs {}\tAcs {}\tNextObs {}\tReward {}\tDones{}".format(np.array(self.observations).shape, np.array(self.actions).shape, np.array(self.next_observations).shape, np.array(self.reward).shape, np.array(self.done).shape))
        # self.log("Actual types: {} {} {} {} {}".format(type(self.observations), type(self.actions), type(self.next_observations), type(self.reward), type(self.done)))

    def _append_step(self):
        if 'Unity' in self.type or 'Particle' in self.type:
            # for ix, buffer in enumerate(self.trajectory_buffers):
            for ix, role in enumerate(self.env.roles):
                '''Order is maintained, each ix is for each Agent Role'''
                exp = list(map(torch.clone, (torch.tensor([self.observations[ix]]),
                                             torch.tensor([self.actions[ix]]),
                                             torch.tensor([self.rewards[ix]]).unsqueeze(dim=-1),
                                             torch.tensor([self.next_observations[ix]]),
                                             torch.tensor([self.dones[ix]], dtype=torch.bool).unsqueeze(dim=-1)
                                             )))
                # buffer.push(exp)
                self.trajectory_buffers[ix].push(exp)
        elif 'Gym' in self.type:
            exp = list(map(torch.clone, (torch.from_numpy(self.observations).unsqueeze(dim=0),
                                         torch.tensor(self.actions).unsqueeze(dim=0),
                                         torch.tensor(self.rewards).reshape(1, 1, 1),
                                         torch.from_numpy(self.next_observations).unsqueeze(dim=0),
                                         torch.tensor(self.dones, dtype=torch.bool).reshape(1, 1, 1)
                                         )))
            self.trajectory_buffers[0].push(exp)
        elif 'RoboCup' in self.type:
            exp = list(map(torch.clone, (torch.tensor([self.observations], dtype=torch.float64),
                                            torch.tensor([self.actions], dtype=torch.float64),
                                            torch.tensor([self.rewards], dtype=torch.float64).unsqueeze(dim=-1),
                                            torch.tensor([self.next_observations], dtype=torch.float64),
                                            torch.tensor([self.dones], dtype=torch.bool).unsqueeze(dim=-1)
                                            )))
            self.trajectory_buffers[0].push(exp)

    def _unity_reshape(self, arr):
        '''Unity reshape of the data - concat all same Role agents trajectories'''
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, dim))

    def _robo_reshape(self, arr):
        arr = np.ascontiguousarray(arr)
        traj, dim = arr.shape
        return np.reshape(arr, (traj, 1, dim))

    def _send_trajectory_numpy(self):
        metrics = self.env.get_metrics(episodic=True)
        self.log(metrics) # print metrics from this end
        if 'Unity' in self.type or 'Particle' in self.type:
            for learner_spec in self.learners_specs:
                self.observations_buffer = []
                self.actions_buffer = []
                self.rewards_buffer = []
                self.next_observations_buffer = []
                self.done_buffer = []
                self.metrics = []
                '''
                    Send 1 MPI message to Learner with the metadata of the trajectory
                    then, send the trajectory data. Note:
                    - trajectory shapes = number of roles x timesteps x (obs or acs dim)
                    - If Agent Roles have different acs/obs dimensions, we may need to split the trajectories in more numpy messages - we could make it dynamic to cover both cases
                '''
                for ix, role in enumerate(learner_spec['roles']):
                    role_ix = self.env.roles.index(role)
                    obs, acs, rew, nobs, don = map(self._unity_reshape, self.trajectory_buffers[role_ix].all_numpy())
                    self.observations_buffer.append(obs)
                    self.actions_buffer.append(acs)
                    self.rewards_buffer.append(rew)
                    self.next_observations_buffer.append(nobs)
                    self.done_buffer.append(don)
                    self.metrics.append(metrics[role_ix]) # accumulate the metrics for each role of this learner

                self.observations_buffer = np.array(self.observations_buffer)
                self.actions_buffer = np.array(self.actions_buffer) # NOTE this will fail if we have 1 learner handling 2 roles with diff acs space
                self.rewards_buffer = np.array(self.rewards_buffer)
                self.next_observations_buffer = np.array(self.next_observations_buffer)
                self.done_buffer = np.array(self.done_buffer)

                trajectory_info = {
                    'env_id': self.id,
                    'role': learner_spec['roles'],
                    'length_index': 1, # index where the Learner can infer the trajectory length
                    'obs_shape': self.observations_buffer.shape,
                    'acs_shape': self.actions_buffer.shape,
                    'rew_shape': self.rewards_buffer.shape,
                    'done_shape': self.done_buffer.shape,
                    'metrics': self.metrics
                }

                # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape, self.rewards_buffer.shape, self.next_observations_buffer.shape, self.done_buffer.shape))
                # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
                # self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))

                learner_ix = learner_spec['id']
                self.learner.send(trajectory_info, dest=learner_ix, tag=Tags.trajectory_info)
                self.learner.Send([self.observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_observations)
                self.learner.Send([self.actions_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_actions)
                self.learner.Send([self.rewards_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_rewards)
                self.learner.Send([self.next_observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_next_observations)
                self.learner.Send([self.done_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_dones)

        elif 'Gym' in self.type:
            '''Assuming 1 Learner and 1 Role'''
            learner_spec = self.learners_specs[0]
            learner_ix = learner_spec['id']
            role_ix = 0

            self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer = map(self._unity_reshape,
                                                                                   self.trajectory_buffers[learner_ix].all_numpy())
            self.metrics = metrics
            trajectory_info = {
                'env_id': self.id,
                'role': learner_spec['roles'],
                'length_index': 0,  # index where the Learner can infer the trajectory length
                'obs_shape': self.observations_buffer.shape,
                'acs_shape': self.actions_buffer.shape,
                'rew_shape': self.rewards_buffer.shape,
                'done_shape': self.done_buffer.shape,
                'metrics': [self.metrics]
            }

            self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape,self.rewards_buffer.shape,self.next_observations_buffer.shape,self.done_buffer.shape))
            # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
            # self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))

        elif 'RoboCup' in self.type:
            for ix in range(self.configs['Evaluation']['agents_per_env']):
                self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer = map(self._robo_reshape, self.trajectory_buffers[0].agent_numpy(ix))

                trajectory_info = {
                    'env_id': self.id,
                    'length': self.env.steps_per_episode,
                    'obs_shape': self.observations_buffer.shape,
                    'acs_shape': self.actions_buffer.shape,
                    'rew_shape': self.rewards_buffer.shape,
                    'done_shape': self.done_buffer.shape,
                    'metrics': self.env.get_metrics(episodic=True)
                }

                # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape,self.rewards_buffer.shape,self.next_observations_buffer.shape,self.done_buffer.shape))
                # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
                #self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))
                # self.log("Trajectory Shapes: Obs {}".format(self.observations_buffer.shape))

                self.learner.send(trajectory_info, dest=self.id, tag=Tags.trajectory_info)
                self.learner.Send([self.observations_buffer, MPI.DOUBLE], dest=self.id, tag=Tags.trajectory_observations)
                self.learner.Send([self.actions_buffer, MPI.DOUBLE], dest=self.id, tag=Tags.trajectory_actions)
                self.learner.Send([self.rewards_buffer, MPI.DOUBLE], dest=self.id, tag=Tags.trajectory_rewards)
                self.learner.Send([self.next_observations_buffer, MPI.DOUBLE], dest=self.id, tag=Tags.trajectory_next_observations)
                self.learner.Send([self.done_buffer, MPI.C_BOOL], dest=self.id, tag=Tags.trajectory_dones)

        self.done_count +=1

        self.reset_buffers()

    def create_buffers(self):
        if 'Unity' in self.type or 'Particle' in self.type:
            '''
                Need a buffer for each Agent Role
                - Agent roles may have different act/obs spaces and number of agent role
                - And each Role may have many agents instances (num_instances_per_env)
                - Order is maintained
            '''
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                              self.env.num_instances_per_role[role],
                                                              self.env.observation_space[role],
                                                              self.env.action_space[role]['acs_space']) \
                                       for i, role in enumerate(self.env.roles) ]
        else:
            '''Gym - has only 1 agent per environment and no roles'''
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                              self.env.num_agents,
                                                              self.env.observation_space,
                                                              self.env.action_space['acs_space'])]

    def reset_buffers(self):
        for buffer in self.trajectory_buffers:
            buffer.reset()

    def _launch_env(self):
        # initiate env from the config
        self.env = self.create_environment()


    def _connect_learners(self):
        self.log("Waiting LearnersSpecs from the MultiEnv")
        self.learners_specs = self.menv.bcast(None, root=0) # Wait until Learners info arrives from MultiEnv
        self.role2learner = self.menv.scatter(None, root=0)
        self.num_learners = len(self.learners_specs)
        # self.log("Got Learners info and will connect with {} learners".format(self.num_learners))
        # Start communication with Learners
        self.learners_port = self.learners_specs[0]['port']
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        self.log("Connected with {} learners on port {}".format(self.num_learners, self.learners_port))
        # Check-in with MultiEnv that we successfully connected with Learner/s
        self.menv.gather(self._get_env_state(), root=0)

    def create_environment(self):
        try:
            self.configs['Environment']['port'] += (self.id * 10)
            self.configs['Environment']['worker_id'] = self.id * 11
        except:
            pass
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
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            'roles': self.env.roles,
            'num_instances_per_env': self.env.num_instances_per_env,
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def print(self, msg, to_print=False):
        text = "{}\t\t\t{}".format(str(self), msg)
        print(text)

    def log(self, msg, to_print=False):
        text = "{}\t\t\t{}".format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<Env(id={})>".format(self.id, MPI.COMM_WORLD.Get_size())

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
    try:
        env = MPIEnv()
    except Exception as e:
        print("Env error:", traceback.format_exc())
    finally:
        terminate_process()

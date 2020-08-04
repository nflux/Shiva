#!/usr/bin/env python
import time, subprocess
import numpy as np
import sys, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import torch

from shiva.core.admin import Admin, logger
from shiva.core.TimeProfiler import TimeProfiler
from shiva.helpers.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.buffers.MultiTensorBuffer import MultiAgentTensorBuffer
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process


class MPIEnv(Environment):
    """ MPI Enabled Environment Wrapper over supported Environments"""
    def __init__(self) -> None:
        
        # for future MPI child abstraction
        self.menv = MPI.COMM_SELF.Get_parent()
        self.id = MPI.COMM_SELF.Get_parent().Get_rank()
        self.info = MPI.Status()
        self.env = None

        # Receive Config from MultiEnv
        self.configs = self.menv.bcast(None, root=0)
        self.menv_id = self.configs['MultiEnv']['id']
        self.configs['Environment']['manual_seed'] += self.menv_id * 100 + self.id
        super(MPIEnv, self).__init__(self.configs)
        self.launch()

    def launch(self) -> None:
        """ Initialize environments, connects to learners, and creates buffers for experience storage.

        Returns:
            None
        """
        self._launch_env()
        # self.log("Received config with {} keys".format(str(len(self.configs.keys()))), verbose_level=1)
        # Check in with MultiEnv
        self.menv.gather(self._get_env_specs(), root=0)

        self._connect_learners()
        self.create_buffers()
        self._receive_new_match()
        # Wait for flag to start running
        start_flag = self.menv.bcast(None, root=0)
        self.log("Start collecting..", verbose_level=1)
        self.run()

    def run(self) -> None:
        """ Checks for trajectories and resets the environment when they are received.

        Returns:
            None
        """
        self.env.reset()
        self.profiler.start(['ExperienceSent'])
        self.is_running = True

        while self.is_running:

            while self.env.start_env(): # give time for the Environment server to be ready (RC specifically)
                self._step_python()
                if not self.is_running:
                    break
                # self._step_numpy()
                self._append_step()
                if self.env.is_done():
                    self._send_trajectory_numpy()
                    self.env.reset()
                # self._reload_match_learners()

        self.close()

    def close(self) -> None:
        """ Disconnects from the multi environment and learner.

        Returns:
            None
        """
        self.log("Started closing", verbose_level=2)
        self.menv.Disconnect()
        self.log("Closed MultiEnv", verbose_level=2)
        self.learner.Disconnect()
        self.log("Closed Learner", verbose_level=2)
        self.log("FULLY CLOSED", verbose_level=1)
        exit(0)

    def _receive_new_match(self):
        self.role2learner_spec = self.menv.recv(None, source=0, tag=Tags.new_agents)
        self.log("Got LearnerSpecs {}".format(self.role2learner_spec), verbose_level=2)
        self.done_count = 0

    def _reload_match_learners(self):
        '''MultiEnv got new matching agents, we receive Learner Specs to send trajectory'''
        if self.menv.Iprobe(source=0, status=self.info):
            self._receive_new_match()
            self.env.reset()
            self.reset_buffers()

    def _step_python(self):
        self.step_count += 1
        self.observations = self.env.get_observations()
        self.menv.gather(self.observations, root=0)
        self.actions = self.menv.scatter(None, root=0)
        if self.actions == False:
            # disconnect signal
            self.is_running = False
        else:
            self.log("Shape Obs {} Act {}".format(np.array(self.observations).shape, np.array(self.actions).shape), verbose_level=3)
            self.log("Obs {} Act {} Rew {}".format(self.observations, self.actions, self.rewards), verbose_level=4)
            self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)

    # def _step_numpy(self):
    #     self.step_count += 1
    #     self.observations = self.env.get_observations()
    #     '''Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
    #                                 --> # of instances of that Agent per EnvWrapper is usually 1, except Unity?
    #     '''
    #     self.observations = np.array(self.observations, dtype=np.float64)
    #     self.menv.Gather([self.observations, MPI.DOUBLE], None, root=0)
    #
    #     if 'Gym' in self.type or 'RoboCup' in self.type:
    #         recv_action = np.zeros((self.env.num_agents, self.env.action_space['acs_space']), dtype=np.float64)
    #         self.menv.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
    #         self.actions = recv_action
    #     elif 'Unity':
    #         self.actions = self.menv.scatter(None, root=0)
    #
    #     self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)
    #     self.log("Obs {} Act {}".format(self.observations, self.actions), verbose_level=3)

    def _append_step(self):
        if 'UnityWrapperEnv012' in self.type or 'ParticleEnv' in self.type:
            # for ix, buffer in enumerate(self.trajectory_buffers):
            for ix, role in enumerate(self.env.roles):
                # self.log(f"Before putting to buffer {self.rewards}")
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
        learners_sent = {spec['id']:False for role, spec in self.role2learner_spec.items()}
        _output_quantity = 0

        if 'UnityWrapperEnv1' in self.type:
            for role, learner_spec in self.role2learner_spec.items():
                learner_ix = learner_spec['id']

                '''No need as we will allow sending multiple messages to a single learner (either because we have multiple agents with same behaviour (Unity-like) or multi agent learner)
                    Potential problem for multiple environments with many within instances:
                        - Order of datas sent from Envs VS order of datas being received by Learner
                '''
                # if learners_sent[learner_ix]:
                #     continue
                # learners_sent[learner_ix] = True

                '''Check if we have any trajectory ready to send'''
                # for role_agent_id in self.env.trajectory_ready_agent_ids[role]:
                while len(self.env.trajectory_ready_agent_ids[role]) > 0:
                    role_agent_id = self.env.trajectory_ready_agent_ids[role].pop()
                    role_ix = self.env.roles.index(role)
                    while len(self.env._ready_trajectories[role][role_agent_id]) > 0:
                        self.done_count += 1
                        _output_quantity += 1

                        # metrics = []
                        # observations_buffer, actions_buffer, rewards_buffer, next_observations_buffer, done_buffer = map(self._unity_reshape, self.trajectory_buffers[role][role_agent_id].all_numpy())
                        observations_buffer, actions_buffer, rewards_buffer, next_observations_buffer, done_buffer, agent_metric = self.env._ready_trajectories[role][role_agent_id].pop()
                        # metrics.append(metrics[role_ix]) # accumulate the metrics for each role of this learner

                        observations_buffer = np.array([observations_buffer])
                        actions_buffer = np.array([actions_buffer]) # NOTE this will fail if we have 1 learner handling 2 roles with diff acs space
                        rewards_buffer = np.array([rewards_buffer])
                        next_observations_buffer = np.array([next_observations_buffer])
                        done_buffer = np.array([done_buffer])

                        trajectory_info = {
                            'env_id': str(self),
                            'role': [role], #learner_spec['roles'],
                            'length_index': 1, # index where the Learner can infer the trajectory length from the shape tuples below
                            'obs_shape': observations_buffer.shape,
                            'acs_shape': actions_buffer.shape,
                            'rew_shape': rewards_buffer.shape,
                            'done_shape': done_buffer.shape,
                            'metrics': [agent_metric] # Learner expects list of metrics, each index for each role
                        }

                        # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape, self.rewards_buffer.shape, self.next_observations_buffer.shape, self.done_buffer.shape))
                        # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
                        # self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))

                        # self.log(f"Obs {observations_buffer.shape} {observations_buffer}")
                        # self.log(f"Acs {actions_buffer}")
                        # self.log(f"Rew {rewards_buffer.shape} {rewards_buffer}")
                        # self.log(f"NextObs {next_observations_buffer}")Œ
                        # self.log(f"Dones {done_buffer}")

                        self.log(f"Sent Learner {learner_ix}, Role: {role}, AgentID {role_agent_id}, Length: {observations_buffer.shape[1]}, TrajRew {rewards_buffer.sum()}", verbose_level=1)
                        self.log(f"{agent_metric}", verbose_level=2)

                        self.learner.send(trajectory_info, dest=learner_ix, tag=Tags.trajectory_info)
                        self.learner.Send([observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_observations)
                        self.learner.Send([actions_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_actions)
                        self.learner.Send([rewards_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_rewards)
                        self.learner.Send([next_observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_next_observations)
                        self.learner.Send([done_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_dones)

        elif 'UnityWrapperEnv012' in self.type or 'Gym' in self.type or 'Particle' in self.type:
            for role, learner_spec in self.role2learner_spec.items():
                learner_ix = learner_spec['id']
                if learners_sent[learner_ix]:
                    continue
                learners_sent[learner_ix] = True

                self.observations_buffer = []
                self.actions_buffer = []
                self.rewards_buffer = []
                self.next_observations_buffer = []
                self.done_buffer = []
                self.metrics = []
                '''
                    First, send 1 MPI message to Learner with the metadata of the trajectory
                    Then, send the trajectory data. Note:
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
                    'env_id': str(self),
                    'role': learner_spec['roles'],
                    'length_index': 1, # index where the Learner can infer the trajectory length from the shape tuples below
                    'obs_shape': self.observations_buffer.shape,
                    'acs_shape': self.actions_buffer.shape,
                    'rew_shape': self.rewards_buffer.shape,
                    'done_shape': self.done_buffer.shape,
                    'metrics': self.metrics
                }

                # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape, self.rewards_buffer.shape, self.next_observations_buffer.shape, self.done_buffer.shape))
                # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
                # self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))

                self.learner.send(trajectory_info, dest=learner_ix, tag=Tags.trajectory_info)
                self.log("Traj sent to {}".format(learner_ix), verbose_level=1)
                self.learner.Send([self.observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_observations)
                self.learner.Send([self.actions_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_actions)
                self.learner.Send([self.rewards_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_rewards)
                self.learner.Send([self.next_observations_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_next_observations)
                self.learner.Send([self.done_buffer, MPI.DOUBLE], dest=learner_ix, tag=Tags.trajectory_dones)

            self.done_count += 1
            _output_quantity = len(learners_sent.keys())
            self.reset_buffers()

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

        if _output_quantity > 0:
            self.profiler.time('ExperienceSent', self.done_count, output_quantity=_output_quantity)
            self.menv.send(_output_quantity, dest=0, tag=Tags.trajectory_info)

    def create_buffers(self) -> None:
        """ Instantiates a tensorbuffer to store the trajectory for each episode.

        Returns:
            None
        """
        if 'UnityWrapperEnv1' in self.type:
            # the append of step data is done at the specific environment implementation
            def nothing(*args, **kwargs):
                return None
            self._append_step = nothing
            self.reset_buffers = self.env.reset_buffers
            self.env.create_buffers()
            self.trajectory_buffers = self.env.trajectory_buffers # change pointer
        elif 'UnityWrapperEnv012' in self.type or 'Particle' in self.type:
            '''
                Need a buffer for each Agent Role
                - Agent roles may have different act/obs spaces and number of agent role
                - And each Role may have many agents instances (num_instances_per_env on Unity)
                - Order is maintained
            '''
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                              self.env.num_instances_per_role[role],
                                                              self.env.observation_space[role],
                                                              sum(self.env.action_space[role]['acs_space'])) \
                                       for i, role in enumerate(self.env.roles) ]
        elif 'Gym' in self.type:
            '''Gym - has only 1 agent per environment and 1 Role'''
            single_role_name = self.env.roles[0]
            self.trajectory_buffers = [ MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                              self.env.num_agents, # = 1
                                                              self.env.observation_space[single_role_name],
                                                              sum(self.env.action_space[single_role_name]['acs_space']))]

    def reset_buffers(self) -> None:
        """ Empties the buffer so that the next trajectory can be stored.
        Returns:
            None
        """
        for buffer in self.trajectory_buffers:
            buffer.reset()

    def _launch_env(self):
        # initiate env from the config
        self.configs['Environment']['port'] += (self.menv_id + 1) * 100 + self.id
        self.configs['Environment']['worker_id'] = (self.menv_id + 1) * 100 + self.id
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        self.env = env_class(self.configs)
        self.log("Launched {}".format(env_class), verbose_level=1)

    def _connect_learners(self):
        self.log("Waiting LearnersSpecs from the MultiEnv", verbose_level=3)
        self.role2learner_spec = self.menv.bcast(None, root=0) # Wait until Learners info arrives from MultiEnv
        self.num_learners = self.configs['MetaLearner']['num_learners']

        # Start communication with Learners
        self.learners_port = self.role2learner_spec[0]['port']['env'][self.menv_id]
        self.log("Trying to connect with Learners", verbose_level=3)
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        self.log("Connected with Learners at port {}".format(self.learners_port), verbose_level=1)
        # Check-in with MultiEnv that we successfully connected with Learner/s
        self.menv.gather(self._get_env_state(), root=0)
        Admin.init(self.configs)
        self.profiler = TimeProfiler(self.configs, Admin.get_learner_url_summary(None, self.role2learner_spec[0]['load_path']), filename_suffix='ME{}-Env{}'.format(self.menv_id, self.id))

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

    def print(self, msg, to_print=False):
        """ Debugging tool to prrint messages and specify what module printed the message.

        Args:
            msg : Message that you want to be printed out
            to_print:

        Returns:
            None
        """
        text = "{}\t\t\t{}".format(str(self), msg)
        print(text)

    def __str__(self):
        return f"<Env(id={self.menv_id}-{self.id}, done_count={self.done_count})>"


if __name__ == "__main__":
    try:
        env = MPIEnv()
    except Exception as e:
        msg = "<Env(id={}) error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
        terminate_process()
    finally:
        pass

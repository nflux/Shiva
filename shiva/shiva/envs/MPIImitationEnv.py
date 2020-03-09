#!/usr/bin/env python
import time, subprocess, socket
import numpy as np
import os, sys, traceback
import random
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import torch

from shiva.core.admin import logger
from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.buffers.TensorBuffer import MultiAgentTensorBuffer, MultiAgentDaggerTensorBuffer
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process

class MPIRoboCupImitationEnv(Environment):

    def __init__(self):
        self.menv = MPI.Comm.Get_parent()
        self.id = self.menv.Get_rank()
        self.launch()
        self.done_count = 0

    def launch(self):
        # Receive Config from MultiEnv
        self.configs = self.menv.bcast(None, root=0)
        super(MPIRoboCupImitationEnv, self).__init__(self.configs)
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self._launch_env()
        # Check in and send single env specs with MultiEnv
        self.menv.gather(self._get_env_specs(), root=0)
        self._connect_learners()
        # Check-in with MultiEnv that successfully connected with Learner
        self.menv.gather(self._get_env_state(), root=0)
        self.create_buffers()
        self._launch_bot_env()
        # Wait for flag to start running
        #self.log("Waiting MultiEnv flag to start")
        start_flag = self.menv.bcast(None, root=0)
        #self.log("Start collecting..")

        self.run()

    # def run(self):
    #     self.env.reset()
    #     while True:
    #         while self.env.start_env():
    #             time.sleep(0.001)
    #             self._super_step_numpy()
    #             self._append_step()
    #             if self.env.is_done():
    #                 self.print(self.env.get_metrics(episodic=True)) # print metrics
    #                 self._send_trajectory_numpy()
    #                 self.log('Episode_count: {}'.format(self.done_count))
    #                 self.env.reset()
    #         # self.close()

    def run(self):
        self.env.reset()
        while True:
            while self.env.start_env():
                self.supervised_run()
                self.dagger_run()
    
    def supervised_run(self):
        for self.super_ep in range(self.supervised_episodes):
            while not self.env.is_done():
                time.sleep(0.001)
                self._super_step_numpy()
                self._super_append_step()
            
            self.print(self.env.get_metrics(episodic=True)) # print metrics
            self._send_super_trajectory_numpy()
            # self.log('Episode_count: {}'.format(self.done_count))
            self.env.reset()
    
    def dagger_run(self):
        while True:
            time.sleep(0.001)
            self._dagger_step_numpy()
            self._dagger_append_step()

            if self.env.is_done():
                self.print(self.env.get_metrics(episodic=True)) # print metrics
                self._send_dagger_trajectory_numpy()
                # self.log('Episode_count: {}'.format(self.done_count))
                self.env.reset()

    def descritize_action(self, action):
        return self.env.descritize_action(action)

    def send_imit_obs_msgs(self):
        self.comm.send(self.env.get_imit_obs_msg())

    def recv_imit_acs_msgs(self):
        acs_msg = self.comm.recv(8192)
        acs_msg = str(acs_msg)[2:].split(' ')[:-1]

        action = np.array(list(map(lambda x: float(x), acs_msg)), dtype=np.float64)
        actions_per_agent = len(action)//self.env.num_agents

        if self.action_level == 'discretized':
            # print('desc action', self.descritize_action(action))
            return np.array([self.descritize_action(action[actions_per_agent*i:actions_per_agent+(actions_per_agent*i)]) for i in range(self.env.num_agents)], dtype=np.float64)
        else:
            return np.array([action[actions_per_agent*i:actions_per_agent+(actions_per_agent*i)] for i in range(self.env.num_agents)], dtype=np.float64)
    
    def _super_step_numpy(self):
        self.observations = self.env.get_observations()
        '''Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
                                    --> # of instances of that Agent per EnvWrapper is usually 1, except Unity?
        '''

        self.send_imit_obs_msgs()
        self.actions = self.recv_imit_acs_msgs()
        # self.log("This is the actions {}".format(self.actions))
        self.observations = np.array(self.observations, dtype=np.float64)

        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions, discrete_select='supervised')

    def _dagger_step_numpy(self):
        self.observations = self.env.get_observations()
        '''Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
                                    --> # of instances of that Agent per EnvWrapper is usually 1, except Unity?
        '''
        self.send_imit_obs_msgs()
        self.bot_action = self.recv_imit_acs_msgs()
        self.observations = np.array(self.observations, dtype=np.float64)
        self.menv.Gather([self.observations, MPI.DOUBLE], None, root=0)

        recv_action = np.zeros((self.env.num_agents, self.env.action_space['acs_space']), dtype=np.float64)
        self.menv.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
        self.actions = recv_action

        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)

    def _super_append_step(self):
        exp = list(map(torch.clone, (torch.tensor([self.observations], dtype=torch.float64),
                                        torch.tensor([self.actions], dtype=torch.float64).unsqueeze(dim=-1),
                                        torch.tensor([self.rewards], dtype=torch.float64).unsqueeze(dim=-1),
                                        torch.tensor([self.next_observations], dtype=torch.float64),
                                        torch.tensor([self.dones], dtype=torch.bool).unsqueeze(dim=-1)
                                        )))
        self.super_trajectory_buffer.push(exp)
    
    def _dagger_append_step(self):
        exp = list(map(torch.clone, (torch.tensor([self.observations], dtype=torch.float64),
                                        torch.tensor([self.actions], dtype=torch.float64),
                                        torch.tensor([self.rewards], dtype=torch.float64).unsqueeze(dim=-1),
                                        torch.tensor([self.next_observations], dtype=torch.float64),
                                        torch.tensor([self.dones], dtype=torch.bool).unsqueeze(dim=-1),
                                        torch.tensor([self.bot_action], dtype=torch.float64).unsqueeze(dim=-1)
                                        )))
        self.dagger_trajectory_buffer.push(exp)

    def _robo_reshape(self, arr):
        arr = np.ascontiguousarray(arr)
        traj, dim = arr.shape
        return np.reshape(arr, (traj, 1, dim))

    def _send_super_trajectory_numpy(self):
        for ix in range(self.num_learners):
            self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer = map(self._robo_reshape, self.super_trajectory_buffer.agent_numpy(ix))

            trajectory_info = {
                'env_id': self.id,
                'length': self.env.steps_per_episode,
                'obs_shape': self.observations_buffer.shape,
                'acs_shape': self.actions_buffer.shape,
                'rew_shape': self.rewards_buffer.shape,
                'done_shape': self.done_buffer.shape,
                'super_done': True if self.super_ep >= (self.supervised_episodes-1) else False,
                'metrics': self.env.get_metrics(episodic=True)
            }

            # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape,self.rewards_buffer.shape,self.next_observations_buffer.shape,self.done_buffer.shape))
            # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
            #self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))
            # self.log("Trajectory Shapes: Obs {}".format(self.observations_buffer.shape))

            self.learner.send(trajectory_info, dest=ix, tag=Tags.trajectory_info)
            self.learner.Send([self.observations_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_observations)
            self.learner.Send([self.actions_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_actions)
            self.learner.Send([self.rewards_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_rewards)
            self.learner.Send([self.next_observations_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_next_observations)
            self.learner.Send([self.done_buffer, MPI.C_BOOL], dest=ix, tag=Tags.trajectory_dones)

        self.done_count +=1

        self.super_reset_buffer()
    
    def _send_dagger_trajectory_numpy(self):
        for ix in range(self.num_learners):
            self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer, self.expert_actions_buffer = map(self._robo_reshape, self.dagger_trajectory_buffer.agent_numpy(ix))

            trajectory_info = {
                'env_id': self.id,
                'length': self.env.steps_per_episode,
                'obs_shape': self.observations_buffer.shape,
                'acs_shape': self.actions_buffer.shape,
                'rew_shape': self.rewards_buffer.shape,
                'done_shape': self.done_buffer.shape,
                'expert_shape': self.expert_actions_buffer.shape,
                'metrics': self.env.get_metrics(episodic=True)
            }

            # self.log("Trajectory Shapes: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.shape, self.actions_buffer.shape,self.rewards_buffer.shape,self.next_observations_buffer.shape,self.done_buffer.shape))
            # self.log("Trajectory Types: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(self.observations_buffer.dtype, self.actions_buffer.dtype, self.rewards_buffer.dtype, self.next_observations_buffer.dtype, self.done_buffer.dtype))
            #self.log("Sending Trajectory Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(self.observations_buffer, self.actions_buffer, self.rewards_buffer, self.next_observations_buffer, self.done_buffer))
            # self.log("Trajectory Shapes: Obs {}".format(self.observations_buffer.shape))

            self.learner.send(trajectory_info, dest=ix, tag=Tags.trajectory_info)
            self.learner.Send([self.observations_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_observations)
            self.learner.Send([self.actions_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_actions)
            self.learner.Send([self.rewards_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_rewards)
            self.learner.Send([self.next_observations_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_next_observations)
            self.learner.Send([self.done_buffer, MPI.C_BOOL], dest=ix, tag=Tags.trajectory_dones)
            self.learner.Send([self.expert_actions_buffer, MPI.DOUBLE], dest=ix, tag=Tags.trajectory_expert_actions)

        self.done_count +=1

        self.dagger_reset_buffer()

    def create_buffers(self):
        self.super_trajectory_buffer = MultiAgentTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                            self.env.num_agents,
                                                            self.env.observation_space,
                                                            1)
        
        self.dagger_trajectory_buffer = MultiAgentDaggerTensorBuffer(self.episode_max_length, self.episode_max_length,
                                                            self.env.num_agents,
                                                            self.env.observation_space,
                                                            self.env.action_space['acs_space'],1)

    def super_reset_buffer(self):
        self.super_trajectory_buffer.reset()
    
    def dagger_reset_buffer(self):
        self.dagger_trajectory_buffer.reset()

    def _launch_env(self):
        # initiate env from the config
        self.env = self.create_environment()
    
    def _launch_bot_env(self):
        cmd = [os.getcwd() + '/shiva/envs/RoboCupBotEnv.py', '-p', str(self.imit_port), '-s', str(self.bot_seed)]
        self.bot_process = subprocess.Popen(cmd, shell=False)

        while True:
            try:
                self.comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.comm.connect(('127.0.0.1', self.imit_port+3))
                break
            except:
                time.sleep(0.1)

    def _connect_learners(self):
        #self.log("Waiting Learners info")
        self.learners_specs = self.menv.bcast(None, root=0) # Wait until Learners info arrives from MultiEnv
        self.num_learners = len(self.learners_specs)
        # self.log("Got Learners info and will connect with {} learners".format(self.num_learners))
        # Start communication with Learners
        self.learners_port = self.learners_specs[0]['port']
        self.supervised_episodes = self.learners_specs[0]['super_ep']
        self.learner = MPI.COMM_WORLD.Connect(self.learners_port, MPI.INFO_NULL)
        #self.log("Connected with {} learners on port {}".format(self.num_learners, self.learners_port))

    def create_environment(self):
        self.configs['Environment']['port'] += (self.id * 10)
        self.configs['Environment']['seed'] = random.randint(0, 10000)
        self.imit_port = self.configs['Environment']['port'] + 3
        self.bot_seed = self.configs['Environment']['seed']
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
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            'agents_group': self.env.agent_groups if hasattr(self.env, 'agent_groups') else ['Agent_0'], # agents names given by the env - needs to be implemented by RoboCup
            'num_instances_per_env': self.env.num_instances_per_env if hasattr(self.env, 'num_instances_per_env') else 1, # Unity case
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }
    
    # Not used for now
    def close_imit(self):
        try:
            self.bot_process.terminate()
            time.sleep(0.1)
            self.bot_process.kill()
        except:
            pass

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
    try:
        env = MPIRoboCupImitationEnv()
    except Exception as e:
        print("Env error:", traceback.format_exc())
    finally:
        terminate_process()

#!/usr/bin/env python
import numpy as np
import sys, time,traceback
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import time


from shiva.core.admin import logger
from shiva.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("shiva")

class MPIEvalEnv(Environment):

    def __init__(self):
        #Comms to communicate with the MPI Evaluation Object
        self.eval = MPI.Comm.Get_parent()
        self.id = self.eval.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from MPI Evaluation Object
        print('Waiting for config')
        self.configs = self.eval.bcast(None, root=0)
        self.log('Received Config')
        super(MPIEvalEnv, self).__init__(self.configs)
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        self._launch_env()
        # self.log('Launched Eval Env')
        # Check in and send single env specs with MPI Evaluation Object
        self.eval.gather(self._get_env_specs(), root=0)
        self.eval.gather(self._get_env_state(), root=0)
        # self.log('Sent specs')
        #self._connect_learners()
        self.create_buffers()
        # Wait for flag to start running
        self.log("Waiting Eval flag to start")
        start_flag = self.eval.bcast(None, root=0)
        self.log("Start collecting..")

        self.run()

    def run(self):
       # self.log("Get here at 45")
        self.env.reset()

        #self.log("Get here at 48")

        while True:
           # time.sleep(0.001)
           # self.log('While True')
            while self.env.start_env():
               # self.log('Env start')
                self._step_numpy()
               # self.log('After step')
                #if self.env.is_done():
                   # self.log('Env is done')
                    #self.env.reset()

                if self.eval.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.clear_buffers):
                    _ = self.eval.recv(None, source=0 , tag=Tags.clear_buffers)
                    self.reset_buffers()
                    print('Buffers have been reset')




                '''Come back to this for emptying old evaluations when a new agent is Loaded

                    if self.eval.bcast(None,root=0):
                    self._clear_buffers()
                    self.debug('Buffer has been cleared')
                    self.env.reset()'''

            self.close()

    def _unity_reshape(self, arr):
        '''Unity reshape of the data - concat all agents trajectories'''
        traj_length, num_agents, dim = arr.shape
        return np.reshape(arr, (traj_length * num_agents, 1, dim))

    def _step_numpy(self):

        self.observations = self.env.get_observations()
        '''Obs Shape = (# of Shiva Agents, # of instances of that Agent per EnvWrapper, Observation dimension)
                                    --> # of instances of that Agent per EnvWrapper is usually 1, except Unity?
        '''
        #self.log("Hello Josh")
        send_obs_buffer = np.array(self.observations, dtype=np.float64)
        self.eval.Gather([send_obs_buffer, MPI.DOUBLE], None, root=0)

        if 'Unity' in self.type:
            self.actions = self.eval.scatter(None, root=0)
            # self.log("Obs {} Act {}".format(self.observations, self.actions))
            self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions.tolist())

            for i in range(len(self.rewards)):
                self.episode_rewards[i,self.reward_idxs[i]] = self.rewards[i]
                if self.dones[i]:
                    self._send_eval_numpy(self.episode_rewards[i,:].sum(),i)
                    self.episode_rewards[i,:].fill(0)
                    self.reward_idxs[i] = 0
                else:
                    self.reward_idxs[i] += 1

        elif 'Gym' in self.type:
            self.actions = self.eval.scatter(None, root=0)
            # self.log("Obs {} Act {}".format(self.observations, self.actions))
            self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions.tolist())
            if self.env.done:
                self._send_eval_numpy(self.env.reward_per_episode,0)
                self.env.reset()

            #self.episode_rewards[self.reward_idxs] = self.rewards
            #if self.dones:
                #self.log('Dones: {}'.format(self.dones))
                #self._send_eval_numpy(self.episode_rewards.sum(),0)
                #self.episode_rewards.fill(0)
                #self.reward_idxs = 0
            #else:
                #self.reward_idxs += 1

        elif 'RoboCup' in self.type:
            # self.log("Getting to 112")
            recv_action = np.zeros((self.env.num_agents, self.env.action_space['acs_space']), dtype=np.float64)
            #self.log("The recv action {}".format(recv_action.shape))
            self.eval.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
            #self.log('Made it to 124')
            self.actions = recv_action
            #self.log("The action is {}".format(self.actions.shape))
            self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)
            #self.log('Made it to 127 {}'.format(self.rewards))

            # for i in range(len(self.rewards)):
            #     self.episode_rewards[i, self.reward_idxs[i]] = self.rewards[i]
            #     if self.dones:
            #         self._send_eval_numpy(self.episode_rewards[i, :].sum(), i)
            #         self.episode_rewards[i, :].fill(0)
            #         self.reward_idxs[i] = 0
            #     else:
            #         self.reward_idxs[i] += 1

            if self.dones:
                self._send_eval_numpy(self.env.env.get_eval_metrics(),0)
                self.env.reset()
                # self.episode_rewards[i, :].fill(0)
                # self.reward_idxs[i] = 0
            # else:
                # self.reward_idxs[i] += 1

                # self.log('Made it to 134')

    def _send_eval_numpy(self, episode_reward, agent_idx):
        '''Numpy approach'''
        self.eval.send(agent_idx, dest=0, tag=Tags.trajectory_info)
        self.log('Eval Reward: {}'.format(episode_reward))
        self.eval.send(episode_reward, dest=0, tag=Tags.trajectory_eval)

    def create_buffers(self):
        if 'Unity' in self.type:
            '''
                Need a buffer for each Agent Group (Brain)
                Agent Groups may have different act/obs spaces and number of agent IDs
                (Unity is a bit different due to the multi-instance per single environment)
            '''
            self.episode_rewards = np.zeros((self.num_agents,self.episode_max_length))
            self.reward_idxs = dict()
            for i in range(self.num_agents): self.reward_idxs[i] = 0

        elif 'Gym' in self.type:
            '''Gym - has only 1 agent per environment and no groups'''
            self.episode_rewards = np.zeros(self.episode_max_length)
            self.reward_idxs = 0
        elif 'RoboCup' in self.type:
            self.episode_rewards = np.zeros((self.num_agents,self.episode_max_length))
            self.reward_idxs = dict()
            for i in range(self.num_agents): self.reward_idxs[i] = 0


    def reset_buffers(self):
        if 'Unity' in self.type:
            '''
                Need a buffer for each Agent Group (Brain)
                Agent Groups may have different act/obs spaces and number of agent IDs
                (Unity is a bit different due to the multi-instance per single environment)
            '''
            self.episode_rewards.fill(0)
            self.reward_idxs = dict()
            for i in range(self.num_agents): self.reward_idxs[i] = 0

        elif 'Gym' in self.type:
            '''Gym - has only 1 agent per environment and no groups'''
            self.episode_rewards.fill(0)
            self.reward_idxs = 0
        elif 'RoboCup' in self.type:
            self.episode_rewards.fill(0)
            self.reward_idxs = dict()
            for i in range(self.num_agents): self.reward_idxs[i] = 0

    def _launch_env(self):
        # initiate env from the config
        self.env = self.create_environment()
        self.num_agents = self.env.num_agents


    def create_environment(self):
        self.configs['Environment']['port'] += 500 +np.random.randint(0,1500)
        self.configs['Environment']['worker_id'] = 100 * (self.id * 22)
        # self.configs['Environment']['rc_log'] = 'rc_eval_log'
        # self.configs['Environment']['server_addr'] = self.eval.Get_attr(MPI.HOST)
        self.configs['Environment']['seed'] = self.id
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        return env_class(self.configs)

    def _get_env_state(self, traj=[]):
        return {
            'metrics': self.env.get_metrics(episodic=True),
            'trajectory': traj
        }

    def _get_env_specs(self):
        return {
            'type': self.type,
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            'agents_group': self.env.agent_groups if hasattr(self.env, 'agent_groups') else ['Agent_0'], # agents names given by the env - needs to be implemented by RoboCup
            'num_instances_per_env': self.env.num_instances_per_env if hasattr(self.env, 'num_instances_per_env') else 1, # Unity case
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'Evaluation Environment {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    try:
        env = MPIEvalEnv()
    except Exception as e:
        print("Eval Env error:", traceback.format_exc())
    finally:
        terminate_process()

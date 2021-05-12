#!/usr/bin/env python
import numpy as np
import sys, time,traceback
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import time

from shiva.helpers.utils.Tags import Tags
from shiva.envs.Environment import Environment
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process
from shiva.core.admin import logger


class MPIEvalEnv(Environment):

    def __init__(self):
        self.eval = MPI.Comm.Get_parent()
        self.id = self.eval.Get_rank()
        self.info = MPI.Status()
        self.launch()

    def launch(self):
        """ Launches environments, creates buffers, tells MPIEvaluation it started.

        Return:
            None
        """
        # Receive Config from MPI Evaluation Object
        self.configs = self.eval.bcast(None, root=0)
        super(MPIEvalEnv, self).__init__(self.configs)
        self._launch_env()
        self.eval.gather(self._get_env_specs(), root=0)

        '''Set function to be run'''
        if 'Gym' in self.type or 'Unity' in self.type or 'ParticleEnv' in self.type or 'MultiAgentGraphEnv' in self.type:
            self.send_evaluations = self._send_eval_roles
        elif 'RoboCup' in self.type:
            self.send_evaluations = self._send_eval_robocup

        self.create_buffers()

        start_flag = self.eval.bcast(None, root=0)
        self.log("Start collecting..", verbose_level=1)
        self.run()

    def run(self):
        """ Starts environments and collects trajectories.

        Returns:
            None
        """
        self.env.reset()

        while True:
            while self.env.start_env():
                self._step_python()
                # self._step_numpy()
                if self.env.is_done(n_episodes=self.configs['Evaluation']['eval_episodes']):
                    self.send_evaluations()
                    self.env.reset(force=True)

                # if self.eval.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.clear_buffers, status=self.info):
                #     _ = self.eval.recv(None, source=self.info.Get_source(), tag=Tags.clear_buffers)
                #     self.reset_buffers()
                #     print('Buffers resets')
            # self.close()

    def _step_python(self):
        self.observations = self.env.get_observations()
        self.eval.gather(self.observations, root=0)
        self.actions = self.eval.scatter(None, root=0)
        self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions)
        self.log("Acs {} Obs {}".format(self.actions, self.observations), verbose_level=3)

    def _step_numpy(self):
        self.observations = self.env.get_observations()
        send_obs_buffer = np.array(self.observations, dtype=np.float64)
        self.eval.Gather([send_obs_buffer, MPI.DOUBLE], None, root=0)

        if 'Gym' in self.type or 'Unity' in self.type or 'ParticleEnv' in self.type or 'MultiAgentGraphEnv' in self.type:
            self.actions = self.eval.scatter(None, root=0)
            self.next_observations, self.rewards, _, _ = self.env.step(self.actions.tolist())
        # elif 'Gym' in self.type:
        #     self.actions = self.eval.scatter(None, root=0)
        #     self.next_observations, self.rewards, self.dones, _ = self.env.step(self.actions.tolist())
        #     # if self.env.done:
        #     #     self._send_eval(self.env.reward_per_episode, 0)
        #     #     self.env.reset()
        elif 'RoboCup' in self.type:
            recv_action = np.zeros((self.env.num_agents, self.env.action_space['acs_space']), dtype=np.float64)
            self.eval.Scatter(None, [recv_action, MPI.DOUBLE], root=0)
            self.actions = recv_action
            self.next_observations, self.rewards, self.dones, _, self.metrics = self.env.step(self.actions, evaluate=True)
            # if self.dones:
            #     self._send_eval(self.metrics, 0)
            #     self.env.reset()
        self.log("Obs {} Act {}".format(self.observations, self.actions), verbose_level=3)

    '''
        Roles Methods
    '''

    def _send_eval_roles(self):
        if 'UnityWrapperEnv1' in self.type:
            # Need to calculate the mean reward for all the simulations within the one Unity environment
            # g.i. 3DBall has 16 simulations within one Unity Environment, so we take the average across 16 agents
            reward_per_episode = {}
            for role in self.env.roles:
                reward_per_episode[role] = []
                for role_agent_id in self.env.trajectory_ready_agent_ids[role]:
                    while len(self.env._ready_trajectories[role][role_agent_id]) > 0:
                        _, _, _, _, _, agent_metric = self.env._ready_trajectories[role][role_agent_id].pop()
                        # self.log(f"Agent_metric {role} {role_agent_id} {agent_metric}")
                        for metric_name, value in agent_metric:
                            if metric_name == 'Reward/Per_Episode':
                                reward_per_episode[role].append(value)
                # print(reward_per_episode)
                reward_per_episode[role] = sum(reward_per_episode[role]) / len(reward_per_episode[role])
            metric = {
                'reward_per_episode': reward_per_episode
            }
        else:
            metric = {
                'reward_per_episode': self.env.get_reward_episode(roles=True)  # dict() that maps role_name->reward
            }
        self.eval.send(metric, dest=0, tag=Tags.trajectory_eval)
        self.log("Sent metrics {}".format(metric), verbose_level=2)

    '''
        Single Agent Methods
    '''

    def _send_eval_robocup(self):
        self._send_eval(self.metrics, 0)

    def _send_eval_gym(self):
        self._send_eval(self.env.reward_per_episode, 0)

    def _send_eval(self, episode_reward, agent_idx):
        self.eval.send(agent_idx, dest=0, tag=Tags.trajectory_eval)
        self.eval.send(episode_reward, dest=0, tag=Tags.trajectory_eval)
        self.log('Eval Reward: {}'.format(episode_reward), verbose_level=2)

    def create_buffers(self):
        """ Creates numpy buffers to store episodic rewards

        Returns:
            None
        """
        if 'Unity' in self.type or 'ParticleEnv' in self.type or 'MultiAgentGraphEnv' in self.type:
            pass
            # self.episode_rewards = np.zeros((len(self.env.roles), self.episode_max_length))
        elif 'Gym' in self.type:
            pass
            # self.episode_rewards = np.zeros(1, self.episode_max_length)
        elif 'RoboCup' in self.type:
            self.episode_rewards = np.zeros((self.num_agents, self.episode_max_length))
            self.reward_idxs = dict()
            for i in range(self.num_agents): self.reward_idxs[i] = 0

    def reset_buffers(self):
        """ Empties the buffers after finishing a trajectory.

        Returns:
            None
        """
        if 'Unity' in self.type or 'MultiAgentGraphEnv' in self.type:
            pass
            # self.episode_rewards.fill(0)
        elif 'Gym' in self.type:
            '''Gym - has only 1 agent per environment and no groups'''
            pass
            # self.episode_rewards.fill(0)
            # self.reward_idxs = 0
        elif 'RoboCup' in self.type:
            self.episode_rewards.fill(0)
            self.reward_idxs = dict()
    #         for i in range(self.num_agents): self.reward_idxs[i] = 0

    def _launch_env(self):
        try:
            self.configs['Environment']['port'] += 500 + np.random.randint(0, 1500)
            self.configs['Environment']['worker_id'] = 1000 * (self.id * 22)
            self.configs['Environment']['render'] = self.configs['Evaluation']['render'] if 'render' in self.configs['Evaluation'] else False
            # self.configs['Environment']['rc_log'] = 'rc_eval_log'
            # self.configs['Environment']['server_addr'] = self.eval.Get_attr(MPI.HOST)
        except:
            pass
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        self.env = env_class(self.configs)
        if 'UnityWrapperEnv1' in self.type:
            self.env.create_buffers()

    def _get_env_specs(self):
        return {
            'type': self.type,
            'id': self.id,
            'observation_space': self.env.get_observation_space(),
            'action_space': self.env.get_action_space(),
            'num_agents': self.env.num_agents,
            'roles': self.env.roles if hasattr(self.env, 'roles') else ['Agent_0'], # agents names given by the env - needs to be implemented by RoboCup
            'num_instances_per_env': self.env.num_instances_per_env if hasattr(self.env, 'num_instances_per_env') else 1, # Unity case
            'learners_port': self.learners_port if hasattr(self, 'learners_port') else False
        }

    def close(self):
        """ Closes the connection with MPIEvaluation

        Returns:
            None
        """
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False, verbose_level=-1):
        """If verbose_level is not given, by default will log
        Args:
            msg: Message to be loged
            to_print: Whether to print it
            verbose_level: When to print it

        Returns:
            None
        """
        if verbose_level <= self.configs['Admin']['log_verbosity']['EvalEnv']:
            text = "{}\t\t\t{}".format(str(self), msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<EvalEnv(id={})>".format(self.id)

    def show_comms(self):
        """ Shows what MPIEvaluation this EvalEnv is connection to.
        Returns:
            None
        """
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    try:
        env = MPIEvalEnv()
    except Exception as e:
        msg = "<EvalEnv(id={})> error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
    finally:
        terminate_process()

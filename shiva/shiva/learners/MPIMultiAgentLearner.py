import sys, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch
import numpy as np
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process
from shiva.learners.Learner import Learner

class MPIMultiAgentLearner(Learner):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPIMultiAgentLearner, self).__init__(self.id, self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)
        # Open Port for Single Environments
        self.port = MPI.Open_port(MPI.INFO_NULL)
        self.log("Open port {}".format(self.port))

        # Set some self attributes from received Config (it should have MultiEnv data!)
        self.MULTI_ENV_FLAG = True
        self.num_envs = self.configs['Environment']['num_instances']
        self.menvs_specs = self.configs['MultiEnv']
        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']
        self.env_specs = self.menvs_specs[0]['env_specs']

        '''Do the Agent selection using my ID (rank)'''
        '''Assuming 1 Agent per Learner!'''
        self.num_agents = 1

        if 'Unity' in self.env_specs['type']:
            self.observation_space = list(self.env_specs['observation_space'].values())[self.id]
            self.action_space = list(self.env_specs['action_space'].values())[self.id]
            self.acs_dim = self.action_space['acs_space']
        elif 'RoboCup' in self.env_specs['type']:
            self.observation_space = self.env_specs['observation_space']
            self.action_space = self.env_specs['action_space']
            self.acs_dim = self.action_space['acs_space'] + self.action_space['param']
        else:
            assert "Not Implemented for Gym"

        self.log("Obs space {} / Action space {}".format(self.observation_space, self.action_space))

        # Check in with Meta
        self.meta.gather(self._get_learner_specs(), root=0)
        # Initialize inter components
        self.alg = self.create_algorithm()
        self.buffer = self.create_buffer()
        self.agents = self.create_agents()
        # make first saving
        Admin.checkpoint(self, checkpoint_num=0, function_only=True, use_temp_folder=True)
        # Connect with MultiEnvs
        self._connect_menvs()

        self.run()

    def run(self, train=True):
        self.log("Waiting for trajectories..")
        self.step_count = 0
        self.done_count = 0
        self.update_num = 0
        self.steps_per_episode = 0
        self.reward_per_episode = 0
        self.train = True

        # '''Used for time calculation'''
        # t0 = time.time()
        # n_episodes = 500
        while self.train:
            # self._receive_trajectory_python_list()
            self._receive_trajectory_numpy()

            # '''Used for time calculation'''
            # if self.done_count == n_episodes:
            #     t1 = time.time()
            #     self.log("Collected {} episodes in {} seconds".format(n_episodes, (t1-t0)))
            #     exit()

            '''Change freely condition when to update'''
            if self.done_count % self.episodes_to_update == 0:
                self.log("Updating at the Learner with done count: {}".format(self.done_count))
                self.alg.update(self.agents[0], self.buffer, self.done_count, episodic=True)
                self.update_num += 1
                self.agents[0].step_count = self.step_count
                self.agents[0].done_count = self.done_count
                # self.log("Sending Agent Step # {} to all MultiEnvs".format(self.step_count))
                Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)
                for ix in range(self.num_menvs):
                    self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

            self.collect_metrics(episodic=True)

            if self.done_count % self.save_checkpoint_episodes == 0:
                Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True)

            '''Send Updated Agents to Meta'''
            self.log("Sending metrics to Meta")
            self.meta.gather(self._get_learner_state(), root=0) # send for evaluation
            '''Check for Evolution Configs'''
            if self.meta.Iprobe(source=0, tag=Tags.evolution):
                evolution_config = self.learners.recv(None, source=0, tag=Tags.evolution)  # block statement
                self.log("Got evolution config!")
            ''''''

    def _receive_trajectory_numpy(self):
        '''Receive trajectory from each single environment in self.envs process group'''
        '''Assuming 1 Agent here, may need to iterate thru all the indexes of the @traj'''

        info = MPI.Status()
        traj_length = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_length, status=info)
        env_source = info.Get_source()

        '''
            Ideas to optimize -> needs some messages that are not multidimensional
                - Concat Observations and Next_Obs into 1 message (the concat won't be multidimensional) 
        '''

        observations = np.zeros([traj_length, self.num_agents, self.observation_space], dtype=np.float64)
        self.envs.Recv([observations, MPI.FLOAT], source=env_source, tag=Tags.trajectory_observations)
        # self.log("Got Obs shape {}".format(observations.shape))

        actions = np.zeros([traj_length, self.num_agents, self.acs_dim], dtype=np.float64)
        self.envs.Recv([actions, MPI.FLOAT], source=env_source, tag=Tags.trajectory_actions)
        # self.log("Got Acs shape {}".format(actions.shape))

        rewards = np.zeros([traj_length, self.num_agents, 1], dtype=np.float64)
        self.envs.Recv([rewards, MPI.FLOAT], source=env_source, tag=Tags.trajectory_rewards)
        # self.log("Got Rewards shape {}".format(rewards.shape))

        next_observations = np.zeros([traj_length, self.num_agents, self.observation_space], dtype=np.float64)
        self.envs.Recv([next_observations, MPI.FLOAT], source=env_source, tag=Tags.trajectory_next_observations)
        # self.log("Got Next Obs shape {}".format(next_observations.shape))

        '''are dones even needed? It's obviously a trajectory...'''
        dones = np.zeros([traj_length, self.num_agents, 1], dtype=np.float64)
        self.envs.Recv([dones, MPI.FLOAT], source=env_source, tag=Tags.trajectory_dones)
        # self.log("Got Dones shape {}".format(dones.shape))

        self.step_count += traj_length
        self.done_count += 1
        self.steps_per_episode = traj_length
        self.reward_per_episode = sum(rewards)

        # self.log("Trajectory shape: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))

        # self.log("{}\n{}\n{}\n{}\n{}".format(type(observations), type(actions), type(rewards), type(next_observations), type(dones)))
        # self.log("{}\n{}\n{}\n{}\n{}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))
        # self.log("{}\n{}\n{}\n{}\n{}".format(observations, actions, rewards, next_observations, dones))

        exp = list(map(torch.clone, (torch.tensor(observations),
                                     torch.tensor(actions),
                                     torch.tensor(rewards),
                                     torch.tensor(next_observations),
                                     torch.tensor(dones, dtype=torch.bool)
                                     )))
  
        self.buffer.push(exp)

    def _connect_menvs(self):
        # Connect with MultiEnv
        self.log("Trying to connect to MultiEnv @ {}".format(self.menv_port))
        self.menv = MPI.COMM_WORLD.Connect(self.menv_port,  MPI.INFO_NULL)
        self.log('Connected with MultiEnv')

        '''Assuming 1 MultiEnv'''
        self.menv.send(self._get_learner_specs(), dest=0, tag=Tags.specs)

        # Accept Single Env Connection
        self.log("Expecting connection from {} Envs @ {}".format(self.num_envs, self.port))
        self.envs = MPI.COMM_WORLD.Accept(self.port)

    def _get_learner_state(self):
        return {
            'train': self.train,
            'num_agents': self.num_agents,
            'update_num': self.update_num,
            'load_path': Admin.get_temp_directory(self),
            'metrics': {}
        }

    def _get_learner_specs(self):
        return {
            'type': 'Learner',
            'id': self.id,
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_temp_directory(self),
        }

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                # ('Reward/Per_Step', self.reward_per_step)
            ]
        else:
            metrics = [

            ]
            try:
                for i, ac in enumerate(self.env_state['buffer'][0][1][-1]):
                    metrics.append(('Agent/Actor_Output_' + str(i), ac))
                metrics += self.env_metrics
            except:
                pass
        return metrics

    def create_agents(self):
        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False)
        else:
            agents = [self.alg.create_agent(ix) for ix in range(self.num_agents)]
        self.log("Agents created: {} of type {}".format(len(agents), type(agents[0])))
        return agents

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Algorithm created of type {}".format(algorithm_class))
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.acs_dim)
        self.log("Buffer created of type {}".format(buffer_class))
        return buffer

    def close(self):
        self.envs.Unpublish_name()
        self.envs.Close_port()
        self.menv.Disconnect()
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()
        MPI.Comm.Disconnect()
        MPI.COMM_WORLD.Abort()

    def log(self, msg, to_print=False):
        text = 'Learner {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(self.menv.Is_inter(), self.menv.Is_intra()))
        self.log("ENV = Inter: {} / Intra: {}".format(self.envs.Is_inter(), self.envs.Is_intra()))


if __name__ == "__main__":
    try:
        l = MPIMultiAgentLearner()
    except Exception as e:
        print("Learner error:", traceback.format_exc())
    finally:
        terminate_process()
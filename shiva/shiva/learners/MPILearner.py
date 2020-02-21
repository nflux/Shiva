import sys, traceback
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch, time
import numpy as np
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process, flat_1d_list
from shiva.learners.Learner import Learner

class MPILearner(Learner):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.scatter(None, root=0)
        # print("Config {}".format(self.configs))
        super(MPILearner, self).__init__(self.id, self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)
        # Open Port for Single Environments
        self.port = MPI.Open_port(MPI.INFO_NULL)
        self.log("Open port {}".format(self.port))

        # Set some self attributes from received Config (it should have MultiEnv data!)
        self.MULTI_ENV_FLAG = True
        self.num_envs = self.configs['Environment']['num_instances']
        '''Assuming all MultiEnvs running have equal Specs in terms of Obs/Acs/Agents'''
        self.menvs_specs = self.configs['MultiEnv']
        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']
        self.env_specs = self.menvs_specs[0]['env_specs']

        self.num_agents = len(self.roles) if hasattr(self, 'roles') else 1

        self.observation_space = self.env_specs['observation_space']
        self.action_space = self.env_specs['action_space']

        # self.log("Got MultiEnvSpecs {}".format(self.menvs_specs))
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

        # '''Used for calculating collection time'''
        # t0 = time.time()
        # n_episodes = 500
        while self.train:
            self._receive_trajectory_numpy()

            # '''Used for calculating collection time'''
            # if self.done_count == n_episodes:
            #     t1 = time.time()
            #     self.log("Collected {} episodes in {} seconds".format(n_episodes, (t1-t0)))
            #     exit()

            if not self.evaluate and len(self.buffer) > self.buffer.batch_size and (self.done_count % self.episodes_to_update == 0):
                self.alg.update(self.agents, self.buffer, self.done_count, episodic=True)
                self.update_num += 1
                for ix in range(len(self.agents)):
                    self.agents[ix].step_count = self.step_count
                    self.agents[ix].done_count = self.done_count
                # self.log("Sending Agent Step # {} to all MultiEnvs".format(self.step_count))
                Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)

                '''No need to send message to MultiEnv'''
                # for ix in range(self.num_menvs):
                #     self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

                if self.done_count % self.save_checkpoint_episodes == 0:
                    Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True)

                '''Send Updated Agents to Meta'''
                # self.log("Sending metrics to Meta")
                self.meta.gather(self._get_learner_state(), root=0) # send for evaluation
                '''Check for Evolution Configs'''
                if self.meta.Iprobe(source=0, tag=Tags.evolution):
                    evolution_config = self.learners.recv(None, source=0, tag=Tags.evolution)  # block statement
                    self.log("Got evolution config!")
                ''''''

            self.log("{}".format([str(a) for a in self.agents]))
            self.collect_metrics(episodic=True)

    def _receive_trajectory_numpy(self):
        '''Receive trajectory from each single environment in self.envs process group'''

        info = MPI.Status()
        self.traj_info = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=info)
        self.log("{}".format(self.traj_info))
        env_source = info.Get_source()

        self.metrics_env = self.traj_info['metrics']
        traj_length_index = self.traj_info['length_index']
        traj_length = self.traj_info['obs_shape'][traj_length_index]
        role = self.traj_info['role']
        assert role == self.roles, "<Learner{}> Got trajectory for {} while we expect for {}".format(self.id, role, self.roles)

        observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
        self.envs.Recv([observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_observations)
        # self.log("Got Obs shape {}".format(observations.shape))

        actions = np.empty(self.traj_info['acs_shape'], dtype=np.float64)
        self.envs.Recv([actions, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_actions)
        # self.log("Got Acs shape {}".format(actions.shape))

        rewards = np.empty(self.traj_info['rew_shape'], dtype=np.float64)
        self.envs.Recv([rewards, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_rewards)
        # self.log("Got Rewards shape {}".format(rewards.shape))

        next_observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
        self.envs.Recv([next_observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_next_observations)
        # self.log("Got Next Obs shape {}".format(next_observations.shape))

        dones = np.empty(self.traj_info['done_shape'], dtype=np.float64)
        self.envs.Recv([dones, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_dones)
        # self.log("Got Dones shape {}".format(dones.shape))

        self.step_count += traj_length
        self.done_count += 1
        self.steps_per_episode = traj_length
        self.reward_per_episode = sum(rewards)

        # self.log("{}\n{}\n{}\n{}\n{}".format(type(observations), type(actions), type(rewards), type(next_observations), type(dones)))
        # self.log("Trajectory shape: Obs {}\t Acs {}\t Reward {}\t NextObs {}\tDones{}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))
        # self.log("Obs {}\n Acs {}\nRew {}\nNextObs {}\nDones {}".format(observations, actions, rewards, next_observations, dones))
        # self.log("From Env received Rew {}\n".format(rewards))

        '''Assuming roles with same acs/obs dimension'''
        exp = list(map(torch.clone, (torch.from_numpy(observations).reshape(traj_length, len(self.roles), observations.shape[-1]),
                                     torch.from_numpy(actions).reshape(traj_length, len(self.roles), actions.shape[-1]),
                                     torch.from_numpy(rewards).reshape(traj_length, len(self.roles), rewards.shape[-1]),
                                     torch.from_numpy(next_observations).reshape(traj_length, len(self.roles), next_observations.shape[-1]),
                                     torch.from_numpy(dones).reshape(traj_length, len(self.roles), dones.shape[-1])
                                     )))

        self.buffer.push(exp)

        # self.close()

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
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles if hasattr(self, 'roles') else False,
            'num_agents': self.num_agents,
            'update_num': self.update_num,
            'load_path': Admin.get_temp_directory(self),
            'metrics': {}
        }

    def _get_learner_specs(self):
        return {
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles if hasattr(self, 'roles') else False,
            'num_agents': self.num_agents,
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_temp_directory(self),
        }

    def get_metrics(self, episodic, agent_id):
        return self.metrics_env[agent_id]

    def create_agents(self):
        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False)
        if hasattr(self, 'roles'):
            self.agents_dict = {role:self.alg.create_agent_of_role(role) for ix, role in enumerate(self.roles)}
            self.log(self.agents_dict)
            agents = list(self.agents_dict.values())
            self.log("{} agents created of type {}".format(len(agents), [str(a) for a in agents]))
        elif self.num_agents == 1:
            agents = [self.alg.create_agent(ix) for ix in range(self.num_agents)]
            self.log("{} agents created of type {}".format(len(agents), str(agents[0])))
        else:
            assert "Some error on creating agents"
        return agents

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        self.configs['Agent']['evaluate'] = self.evaluate
        self.configs['Algorithm']['roles'] = self.roles
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Algorithm created of type {}".format(algorithm_class ))
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if hasattr(self, 'roles'):
            '''Assuming roles with same obs/acs dim'''
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space[self.roles[0]], self.action_space[self.roles[0]]['acs_space'])
        else:
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.log("Buffer created of type {}".format(buffer_class))
        return buffer

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

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
        l = MPILearner()
    except Exception as e:
        print("Learner error:", traceback.format_exc())
    finally:
        terminate_process()
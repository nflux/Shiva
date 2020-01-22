import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch
import logging
from mpi4py import MPI

from shiva.core.admin import Admin
from shiva.helpers.config_handler import load_class
from shiva.learners.Learner import Learner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

class MPILearner(Learner):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.bcast(None, root=0)
        super(MPILearner, self).__init__(self.id, self.configs)
        self.debug("Received config with {} keys".format(len(self.configs.keys())))
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)
        # Open Port for Single Environments
        self.port = MPI.Open_port(MPI.INFO_NULL)
        self.debug("Open port {}".format(self.port))

        # Set some self attributes from received Config (it should have MultiEnv data!)
        self.num_envs = self.configs['Environment']['num_instances']
        self.num_agents = 1 # self.configs['Environment']['num_agents']
        self.menvs_specs = self.configs['MultiEnv']
        self.num_menvs = len(self.menvs_specs)
        # assuming 1 MultiEnv here
        self.menv_port = self.menvs_specs[0]['port']
        self.observation_space = self.menvs_specs[0]['env_specs']['observation_space']
        self.action_space = self.menvs_specs[0]['env_specs']['action_space']

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

        self.train = True
        self.run()

    def run(self, train=True):
        self.debug("Waiting for trajectories..")
        self.step_count = 0
        self.done_count = 0
        self.update_num = 0
        while self.train:
            traj = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=7) # blocking operation until all environments sent at least 1 trajectory

            self.done_count += 1

            '''Assuming 1 Agent here, may need to iterate thru all the indexes of the @traj'''
            agent_ix = 0
            observations, actions, rewards, next_observations, dones = traj[agent_ix]

            # self.debug("{}\n{}\n{}\n{}\n{}".format(type(observations), type(actions), type(rewards), type(next_observations), type(dones)))
            # self.debug("{}\n{}\n{}\n{}\n{}".format(observations, actions, rewards, next_observations, dones))

            self.step_count += len(observations)
            exp = list(map(torch.clone, (torch.tensor(observations), torch.tensor(actions), torch.tensor(rewards).reshape(-1, 1), torch.tensor(next_observations), torch.tensor(dones, dtype=torch.bool).reshape(-1, 1))))
            self.buffer.push(exp)

            '''Change freely condition when to update'''
            if self.done_count % self.num_envs == 0:
                self.alg.update(self.agents[0], self.buffer, self.done_count, episodic=True)
                self.update_num += 1
                self.agents[0].step_count = self.step_count
                self.agents[0].done_count = self.done_count
                # self.debug("Sending Agent Step # {} to all MultiEnvs".format(self.step_count))
                Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)
                for ix in range(self.num_menvs):
                    self.menv.send(self._get_learner_state(), dest=ix, tag=10)

            if self.update_num % self.save_checkpoint_episodes == 0:
                Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True)
                # self.collect_metrics()

            '''Send Updated Agents to Meta'''
            # self.debug("Sending metrics to Meta")
            self.meta.gather(self._get_learner_state(), root=0) # send for evaluation
            '''Check for Evolution Configs'''
            if self.meta.Iprobe(source=0, tag=1):
                evolution_config = self.learners.recv(None, source=0, tag=11)  # block statement
                self.debug("Got evolution config!")
            ''''''

    def _get_trajectory(self):
        for ix in range(self.num_envs):
            probe = self.envs.probe(source=ix, tag=7)
            print(probe)

    def _connect_menvs(self):
        # Connect with MultiEnv
        self.debug("Trying to connect to MultiEnv @ {}".format(self.menv_port))
        self.menv = MPI.COMM_WORLD.Connect(self.menv_port,  MPI.INFO_NULL)
        self.debug('Connected with MultiEnv')

        # Check in with MultiEnv # 0, there's only 1 for now
        self.menv.send(self._get_learner_specs(), dest=0, tag=0)

        # Accept Single Env Connection
        self.debug("Expecting connection from {} Envs @ {}".format(self.num_envs, self.port))
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

    def create_agents(self):
        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False)
        else:
            agents = [self.alg.create_agent(ix) for ix in range(self.num_agents)]
        self.debug("Agents created: {} of type {}".format(len(agents), type(agents[0])))
        return agents

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.debug("Algorithm created of type {}".format(algorithm_class))
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.debug("Buffer created of type {}".format(buffer_class))
        return buffer

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def debug(self, msg, to_print=False):
        text = 'Learner {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.debug(text)
        if to_print or self.configs['Admin']['debug']:
            print(text)

    def show_comms(self):
        self.debug("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.debug("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.debug("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.debug("MENV = Inter: {} / Intra: {}".format(self.menv.Is_inter(), self.menv.Is_intra()))
        self.debug("ENV = Inter: {} / Intra: {}".format(self.envs.Is_inter(), self.envs.Is_intra()))


if __name__ == "__main__":
    l = MPILearner()
import sys, traceback, os
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch
import numpy as np
import uuid
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
import shiva.helpers.file_handler as fh
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process
from shiva.learners.Learner import Learner

class MPILearner(Learner):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.id = self.meta.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from Meta
        self.configs = self.meta.scatter(None, root=0)
        super(MPILearner, self).__init__(self.id, self.configs)
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)
        # Open Port for Single Environments
        self.port = MPI.Open_port(MPI.INFO_NULL)
        #self.log("Open port {}".format(str(self.port)))

        # Set some self attributes from received Config (it should have MultiEnv data!)
        self.MULTI_ENV_FLAG = True
        self.num_envs = self.configs['Environment']['num_instances']
        '''Assuming all MultiEnvs running have equal Specs in terms of Obs/Acs/Agents'''
        self.menvs_specs = self.configs['MultiEnv']
        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']
        self.env_specs = self.menvs_specs[0]['env_specs']
        '''Do the Agent selection using my ID (rank)'''
        '''Assuming 1 Agent per Learner!'''

        try:
            # self.observation_space = self.env_specs['observation_space'][self.config['Learner']['group']]
            # self.action_space = self.env_specs['action_space'][self.config['Learner']['group']]
            self.observation_space = list(self.env_specs['observation_space'].values())[self.id]
            self.action_space = list(self.env_specs['action_space'].values())[self.id]
        except:
            self.observation_space = self.env_specs['observation_space']
            self.action_space = self.env_specs['action_space']

        # self.log("Got MultiEnvSpecs {}".format(self.menvs_specs))
        #self.log("Obs space {} / Action space {}".format(self.observation_space, self.action_space))
        # Check in with Meta
        self.meta.gather(self._get_learner_specs(), root=0)

        # Initialize inter components
        self.alg = self.create_algorithm()
        self.buffer = self.create_buffer()
        self.agents, self.agent_ids = self.create_agents()
        if self.pbt:
            self.create_pbt_dirs()
            self.save_pbt_agents()
            for agent in self.agents:
                agent.save(self.eval_path+'Agent_'+str(agent.id),0)

        self.meta.gather(self.agent_ids,root=0)

        self.evolution_checks = 1
        # make first saving
        Admin.checkpoint(self, checkpoint_num=0, function_only=True, use_temp_folder=True)
        # Connect with MultiEnvs
        self._connect_menvs()
        self.run()

    def run(self, train=True):
        #self.log("Waiting for trajectories..")
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
            # self._receive_trajectory_python_list()
            self._receive_trajectory_numpy()

            # '''Used for calculating collection time'''
            # if self.done_count == n_episodes:
            #     t1 = time.time()
            #     self.log("Collected {} episodes in {} seconds".format(n_episodes, (t1-t0)))
            #     exit()

            if not self.evaluate:
                '''Change freely condition when to update'''
                if self.done_count % self.episodes_to_update == 0:
                    self.alg.update(self.agents[0], self.buffer, self.done_count, episodic=True)
                    self.update_num += 1
                    self.agents[0].step_count = self.step_count
                    self.agents[0].done_count = self.done_count
                    # self.log("Sending Agent Step # {} to all MultiEnvs".format(self.step_count))
                    Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)
                    for ix in range(self.num_menvs):
                        self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

                if self.done_count % self.save_checkpoint_episodes == 0:
                    Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True)

            if self.done_count % self.evolution_episodes == 0:
                print('Requesting evolution config')
                self.meta.send(self.agent_ids, dest=0, tag=Tags.evolution) # send for evaluation
                for agent in self.agents:
                    self.evolution_config = self.meta.recv(None, source=0, tag=Tags.evolution_config)  # block statement
                    print('Received evolution config')
                    if self.evolution_config['evolution'] == False:
                        continue
                    setattr(self, 'exploitation', self.evolution_config['exploitation'])
                    setattr(self, 'exploration', self.evolution_config['exploration'])
                    print('Starting Evolution')
                    self.exploitation = getattr(self, self.exploitation)
                    self.exploration = getattr(self, self.exploration)
                    self.exploitation(agent,self.evolution_config)
                    self.exploration(agent)
                    print('Evolution Complete\n\n\n\n\n')




                self.log("Got evolution config!")
                ''''''

            self.collect_metrics(episodic=True)

    def _receive_trajectory_numpy(self):
        '''Receive trajectory from each single environment in self.envs process group'''
        '''Assuming 1 Agent here (no support for MADDPG), may need to iterate thru all the indexes of the @traj'''

        info = MPI.Status()
        self.traj_info = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=info)
        env_source = info.Get_source()

        '''Assuming 1 Agent here'''
        self.metrics_env = self.traj_info['metrics']
        traj_length = self.traj_info['length']

        #self.log("{}".format(self.traj_info))

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

        exp = list(map(torch.clone, (torch.from_numpy(observations),
                                     torch.from_numpy(actions),
                                     torch.from_numpy(rewards),
                                     torch.from_numpy(next_observations),
                                     torch.from_numpy(dones)
                                     )))
        self.buffer.push(exp)

        # self.close()

    def _connect_menvs(self):
        # Connect with MultiEnv
        #self.log("Trying to connect to MultiEnv @ {}".format(self.menv_port))
        self.menv = MPI.COMM_WORLD.Connect(self.menv_port,  MPI.INFO_NULL)
        #self.log('Connected with MultiEnv')

        '''Assuming 1 MultiEnv'''
        self.menv.send(self._get_learner_specs(), dest=0, tag=Tags.specs)

        # Accept Single Env Connection
        #self.log("Expecting connection from {} Envs @ {}".format(self.num_envs, self.port))
        self.envs = MPI.COMM_WORLD.Accept(self.port)

    def _get_learner_state(self):
        return {
            'evaluate': self.evaluate,
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
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_temp_directory(self),
        }

    def get_metrics(self, episodic=False):
        '''Assuming 1 agent here'''
        return self.metrics_env

    def create_agents(self):
        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False)
        else:
            #self.start_agent_idx = self.num_agents * self.id
            #self.end_agent_idx = self.start_agent_idx + self.num_agents
            #agents = [self.alg.create_agent(ix) for ix in np.arange(self.start_agent_idx,self.end_agent_idx)]
            agents = [self.alg.create_agent(uuid.uuid4().int) for i in range (self.num_agents)]
            agent_ids = [agent.id for agent in agents]
        self.log("Agents created: {} of type {}".format(len(agents), type(agents[0])))
        return agents, agent_ids

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Algorithm created of type {}".format(algorithm_class))
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.log("Buffer created of type {}".format(buffer_class))
        return buffer

    def save_pbt_agents(self):
        for agent in self.agents:
            agent_path = self.eval_path+'Agent_'+str(agent.id)
            agent.save(agent_path,0)
            fh.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))

    def create_pbt_dirs(self):
        for agent in self.agents:
            if not os.path.isdir(self.eval_path+'Agent_'+str(agent.id)):
                agent_dir = self.eval_path+'Agent_'+str(agent.id)
                os.mkdir(agent_dir)

    def welch_T_Test(self,evals,evo_evals):
        t,p = stats.ttest_ind(evals, evo_evals, equal_var=False)
        return p < self.p_value

    def t_test(self,agent,evo_config):
        if evo_config['ranking'] < evo_config['evo_ranking']:
            evals = np.load(this.eval_path+'Agent_'+str(evo_config['agent']+'/episode_evaluations'))
            evo_evals = np.load(this.eval_path+'Agent_'+str(evo_config['evo_agent']+'/episode_evaluations'))
            if welch_T_Test(evals,evo_evals):
                path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
                evo_agent = Admin._load_agents(path)[0]
                agent.copy_weights(evo_agent)

    def truncation(self,agent,evo_config):
        print('Truncating')
        path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
        evo_agent = Admin._load_agents(path)[0]
        agent.copy_weights(evo_agent)
        print('Truncated')

    def perturb(self,agent):
        perturb_factor = np.random.choice([0.8,1.2])
        agent.perturb_hyperparameters(perturb_factor)

    def resample(self,agent):
        agent.resample_hyperparameters()

    def exploitation(self):
        pass

    def exploration(self):
        pass

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

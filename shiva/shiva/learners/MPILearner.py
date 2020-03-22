import sys, traceback, os, time, pickle
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch, time
import numpy as np
from scipy import stats
from mpi4py import MPI

from shiva.utils.Tags import Tags
from shiva.core.admin import Admin, logger
import shiva.helpers.file_handler as fh
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
        self.set_default_configs()
        super(MPILearner, self).__init__(self.id, self.configs)
        self._connect_io_handler()
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)
        #print('This is my Admin object: {}'.format(Admin))
        # Open Port for Single Environments
        self.port = MPI.Open_port(MPI.INFO_NULL)
        #self.log("Open port {}".format(str(self.port)))

        # Set some self attributes from received Config (it should have MultiEnv data!)
        self.MULTI_ENV_FLAG = True
        self.num_envs = self.configs['Environment']['num_envs']
        '''Assuming all MultiEnvs running have equal Specs in terms of Obs/Acs/Agents'''
        self.menvs_specs = self.configs['MultiEnv']
        self.env_specs = self.menvs_specs[0]['env_specs']

        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']

        if not hasattr(self, 'roles'):
            self.roles = self.env_specs['roles'] # take all roles
        self.num_agents = len(self.roles)

        # if 'Unity' in self.env_specs['type']:
        #     # self.observation_space = self.env_specs['observation_space'][self.config['Learner']['group']]
        #     # self.action_space = self.env_specs['action_space'][self.config['Learner']['group']]
        #     # self.log("Env specs {} id {}".format(self.env_specs['observation_space'], self.id))
        #     self.observation_space = list(self.env_specs['observation_space'].values())[0]
        #     self.action_space = list(self.env_specs['action_space'].values())[0]
        # elif 'Gym' in self.env_specs['type']:
        #     self.observation_space = self.env_specs['observation_space']
        #     self.action_space = self.env_specs['action_space']
        # elif 'RoboCup' in self.env_specs['type']:
        #     self.observation_space = self.env_specs['observation_space']
        #     self.action_space = self.env_specs['action_space']
        '''Seems that all the If conditions were solved'''
        self.observation_space = self.env_specs['observation_space']
        self.action_space = self.env_specs['action_space']

        # self.log("Got MultiEnvSpecs {}".format(self.menvs_specs))
        self.log("Obs space {} / Action space {}".format(self.observation_space, self.action_space))

        self.t_test_config = dict()

        # Check in with Meta
        self.meta.gather(self._get_learner_specs(), root=0)

        # Initialize inter components
        self.alg = self.create_algorithm()
        self.buffer = self.create_buffer()
        self.agents, self.agent_ids = self.create_agents()

        if self.pbt:
            self.create_pbt_dirs()
            self.log('Sending IO save pbt agents request')
            self.save_pbt_agents()
        self.meta.gather(self.agent_ids, root=0)

        # make first saving
        Admin.checkpoint(self, checkpoint_num=0, function_only=True, use_temp_folder=True)
        #self._io_checkpoint(checkpoint_num=0,function_only=True,use_temp_folder=True)
        # Connect with MultiEnvs
        self._connect_menvs()
        self.run()

    def run(self):
        self.step_count = 0
        self.done_count = 0
        self.num_updates = 0
        self.steps_per_episode = 0
        self.reward_per_episode = 0

        # '''Used for calculating collection time'''
        # t0 = time.time()
        # n_episodes = 500
        while True:
            self.receive_trajectory_numpy()

            self.log('Episodes collected: {}'.format(self.done_count))
            # '''Used for calculating collection time'''
            # if self.done_count == n_episodes:
            #     t1 = time.time()
            #     self.log("Collected {} episodes in {} seconds".format(n_episodes, (t1-t0)))
            #     exit()
            self.run_updates()
            self.run_evolution()
            self.collect_metrics(episodic=True)

    def receive_trajectory_numpy(self):
        '''Receive trajectory from each single environment in self.envs process group'''
        info = MPI.Status()
        if self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=info):
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

    def run_updates(self):
        '''Training'''
        if not self.evaluate and len(self.buffer) > self.buffer.batch_size and (self.done_count % self.episodes_to_update == 0):
            self.alg.update(self.agents, self.buffer, self.done_count, episodic=True)
            self.num_updates = self.alg.get_num_updates()
            for ix in range(len(self.agents)):
                self.agents[ix].step_count = self.step_count
                self.agents[ix].done_count = self.done_count
                self.agents[ix].num_updates = self.num_updates
            '''Save latest updated agent in temp folder for MultiEnv to load'''
            self._io_checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)

            '''No need to send message to MultiEnv for now'''
            # for ix in range(self.num_menvs):
            #     self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

            if self.pbt:
                self._io_save_pbt_agents()

            '''Check point purposes only'''
            if self.done_count % self.save_checkpoint_episodes == 0:
                self._io_checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=False)

    def run_evolution(self):
        '''Evolution'''
        if self.pbt:
            if self.done_count % self.evolution_episodes == 0 and (self.done_count >= self.initial_evolution_episodes):
                self.meta.send(self.agent_ids, dest=0, tag=Tags.evolution) # send for evaluation

            info = MPI.Status()
            if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution_config, status=info):
                meta_source = info.Get_source()
                self.log('Starting Evolution for {} agents'.format(len(self.agents)))
                for agent in self.agents:
                    self.evolution_config = self.meta.recv(None, source=meta_source, tag=Tags.evolution_config)  # block statement
                    self.log('Received EvolutionConfig for {}'.format(str(agent)))
                    if self.evolution_config['evolution'] == False:
                        continue
                    setattr(self, 'exploitation', getattr(self, self.evolution_config['exploitation']))
                    setattr(self, 'exploration', getattr(self, self.evolution_config['exploration']))
                    self.exploitation(agent,self.evolution_config)
                    self.exploration(agent)

                '''No need to send message to MultiEnv for now'''
                # for ix in range(self.num_menvs):
                #     self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

                self.log('Evolution Completed for {} agents'.format(len(self.agents)))

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
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles,
            'num_agents': self.num_agents,
            'num_updates': self.num_updates,
            'load_path': Admin.get_temp_directory(self),
            'metrics': {}
        }

    def _get_learner_specs(self):
        return {
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles,
            'num_agents': self.num_agents,
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_temp_directory(self),
        }

    def get_metrics(self, episodic, agent_id):
        return self.metrics_env[agent_id]

    def create_agents(self):
        assert hasattr(self, 'num_agents') and self.num_agents > 0, 'Learner num_agent not specified, got {}'.format(self.num_agents)
        self.start_agent_idx = self.num_agents * self.id * 10
        self.agent_ids = np.arange(self.start_agent_idx, self.start_agent_idx + self.num_agents)

        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False)
            self.agent_ids = [a.id for a in agents]
            self.log("{} agents loaded".format([str(a) for a in agents]))
        elif hasattr(self, 'roles') and len(self.roles) > 0:
            self.agents_dict = {role:self.alg.create_agent_of_role(self.agent_ids[ix], role) for ix, role in enumerate(self.roles)}
            self.log(self.agents_dict)
            agents = list(self.agents_dict.values())
            self.log("{} agents created: {}".format(len(agents), [str(a) for a in agents]))
        else:
            agents = [self.alg.create_agent(ix) for ix in self.agent_ids]
            self.log("{} agents created: {}".format(len(agents), [str(a) for a in agents]))
        self.metrics_env = {agent.id:[] for agent in agents}

        for _agent in agents:
            _agent.evaluate = self.evaluate
        return agents, self.agent_ids

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        self.configs['Algorithm']['roles'] = self.roles if hasattr(self, 'roles') else []
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Algorithm created of type {}".format(algorithm_class))
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if type(self.observation_space) == dict:
            '''Assuming roles with same obs/acs dim'''
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space[self.roles[0]], self.action_space[self.roles[0]]['acs_space'])
        else:
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.log("Buffer created of type {}".format(buffer_class))
        return buffer

    def _connect_io_handler(self):
        # self.log('Sending IO Connection Request')
        self.io = MPI.COMM_WORLD.Connect(self.learners_io_port, MPI.INFO_NULL)
        self.log('Connected with IOHandler')
        self.io_request = dict()
        self.io_pbt_request = dict()
        self.io_pbt_request['path'] = self.eval_path+'Agent_'

    def _io_checkpoint(self, checkpoint_num, function_only, use_temp_folder):
        self.io.send(True, dest=0, tag=Tags.io_learner_request)
        _ = self.io.recv(None, source=0, tag=Tags.io_learner_request)
        Admin.checkpoint(self, checkpoint_num=checkpoint_num, function_only=function_only, use_temp_folder=use_temp_folder)
        self.io.send(True, dest=0, tag=Tags.io_learner_request)
        #self.io_request['learner'] = self.id
        #self.io_request['checkpoint_num'] = checkpoint_num
        #self.io_request['function_only'] = function_only
        #self.io_request['use_temp_folder'] = use_temp_folder
        #self.io_request['agents'] = self.agents
        #self.io_request['checkpoint_path'] = Admin.new_checkpoint_dir(self,checkpoint_num)
        #self.io_request['agent_dir'] = [Admin.get_new_agent_dir(self,agent) for agent in self.agents]
        #self.io.send(self.io_request,dest=0,tag=Tags.io_checkpoint_save)

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

    def _io_save_pbt_agents(self):
        self.io.send(True, dest=0, tag=Tags.io_learner_request)
        _ = self.io.recv(None, source = 0, tag=Tags.io_learner_request)
        self.save_pbt_agents()
        self.io.send(True, dest=0, tag=Tags.io_learner_request)

    def welch_T_Test(self,evals,evo_evals):
        if 'RoboCup' in self.configs['Environment']['type']:
            return True
        else:
            t,p = stats.ttest_ind(evals, evo_evals, equal_var=False)
            return p < self.p_value

    def _t_test(self,agent,evo_config):
        # print('Starting t_test')
        if evo_config['ranking'] > evo_config['evo_ranking']:
            # print('Ranking > Evo_Ranking')
            evals = np.load(self.eval_path+'Agent_'+str(evo_config['agent'])+'/episode_evaluations.npy')
            evo_evals = np.load(self.eval_path+'Agent_'+str(evo_config['evo_agent'])+'/episode_evaluations.npy')
            if self.welch_T_Test(evals,evo_evals):
                # print('Welch Passed')
                path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
                evo_agent = Admin._load_agents(path)[0]
                agent.copy_weights(evo_agent)
        # print('Finished t_test')

    def t_test(self,agent,evo_config):
        if evo_config['ranking'] > evo_config['evo_ranking']:
            #self.t_test_config['evals_path'] = self.eval_path+'Agent_'+str(evo_config['agent'])+'/episode_evaluations.npy'
            #self.t_test_config['evo_evals_path'] = self.eval_path+'Agent_'+str(evo_config['evo_agent'])+'/episode_evaluations.npy'
            #self.t_test_config['evo_agent_path'] = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
            #self.io.send(self.t_test_config,dest=0,tag=Tags.io_evals_load)
            #self.evo_evals = self.io.recv(None,source=MPI.ANY_SOURCE, tag=Tags.io_evals_load)
            self.io.send(True, dest=0, tag=Tags.io_learner_request)
            _ = self.io.recv(None, source = 0, tag=Tags.io_learner_request)
            path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
            if 'RoboCup' in self.configs['Environment']['type']:
                with open(self.eval_path+'Agent_'+str(evo_config['agent'])+'/episode_evaluations.data','rb') as file_handler:
                    evals = np.array(pickle.load(file_handler))
                with open(self.eval_path+'Agent_'+str(evo_config['evo_agent'])+'/episode_evaluations.data','rb') as file_handler:
                    evo_evals = np.array(pickle.load(file_handler))
            else :
                evals = np.load(self.eval_path+'Agent_'+str(evo_config['agent'])+'/episode_evaluations.npy')
                evo_evals = np.load(self.eval_path+'Agent_'+str(evo_config['evo_agent'])+'/episode_evaluations.npy')
            #path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
            evo_agent = Admin._load_agents(path)[0]
            self.io.send(True, dest=0, tag=Tags.io_learner_request)
            if self.welch_T_Test(evals,evo_evals):
                agent.copy_weights(evo_agent)

    def _truncation(self,agent,evo_config):
        # print('Truncating')
        path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
        evo_agent = Admin._load_agents(path)[0]
        agent.copy_weights(evo_agent)
        # print('Truncated')

    def truncation(self,agent,evo_config):
        path = self.eval_path+'Agent_'+str(evo_config['evo_agent'])
        self.io.send(True, dest=0, tag=Tags.io_learner_request)
        _ = self.io.recv(None, source = 0, tag=Tags.io_learner_request)
        evo_agent = Admin._load_agents(path)[0]
        self.io.send(True, dest=0, tag=Tags.io_learner_request)
        agent.copy_weights(evo_agent)

    def perturb(self,agent):
        # print('Pertubing')
        perturb_factor = np.random.choice([0.8,1.2])
        agent.perturb_hyperparameters(perturb_factor)
        # print('Finished Pertubing')

    def resample(self,agent):
        # print('Resampling')
        agent.resample_hyperparameters()

    def exploitation(self):
        pass

    def exploration(self):
        pass

    def set_default_configs(self):
        assert 'Learner' in self.configs, 'No Learner config found on {}'.format(self.configs)
        if not hasattr(self.configs['Learner'], 'evaluate'):
            self.configs['Learner']['evaluate'] = False

    def close(self):
        comm = MPI.Comm.Get_parent()
        self.envs.Disconnect()
        self.menv.Disconnet()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = '{}\t{}'.format(str(self), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<Learner(id={})>".format(self.id)

    def show_comms(self):
        self.log("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.log("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.log("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))
        self.log("MENV = Inter: {} / Intra: {}".format(self.menv.Is_inter(), self.menv.Is_intra()))
        self.log("ENV = Inter: {} / Intra: {}".format(self.envs.Is_inter(), self.envs.Is_intra()))
        self.log("MEVAL = Inter: {} / Intra: {}".format(self.meval.Is_inter(), self.meval.Is_intra()))


if __name__ == "__main__":
    try:
        l = MPILearner()
    except Exception as e:
        print("Learner error:", traceback.format_exc())
    finally:
        terminate_process()

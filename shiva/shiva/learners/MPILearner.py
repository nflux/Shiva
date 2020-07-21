import sys, traceback, os, time, pickle
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch, time
import numpy as np
from scipy import stats
from collections import deque
from mpi4py import MPI

from shiva.helpers.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.core.IOHandler import get_io_stub
from shiva.helpers.config_handler import load_class
from shiva.helpers.misc import terminate_process, flat_1d_list
from shiva.learners.Learner import Learner

class MPILearner(Learner):

    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    MULTI_ENV_FLAG = True  # only being used for get_metrics function - to be handled differently at some point

    def __init__(self):
        # Receive Config from Meta
        self.configs = self.meta.scatter(None, root=0)
        self.set_default_configs()
        super(MPILearner, self).__init__(self.id, self.configs)
        self.log("Received config with {} keys".format(len(self.configs.keys())), verbose_level=1)
        self.log("Received config {}".format(self.configs), verbose_level=3)
        self.launch()

    def launch(self):
        self._connect_io_handler()

        '''Process Configs'''
        # Assuming all Environments running are of the same type
        self.menvs_specs = self.configs['MultiEnv']
        self.env_specs = self.menvs_specs[0]['env_specs']
        self.num_envs = self.configs['Environment']['num_envs']
        self.num_menvs = len(self.menvs_specs)
        self.menv_port = self.menvs_specs[0]['port']

        # Create Port for each single Environment process group
        self.port = {'env': []}
        for i in range(self.num_menvs):
            self.port['env'].append(MPI.Open_port(MPI.INFO_NULL))

        if not hasattr(self, 'roles'):
            self.roles = self.env_specs['roles'] # take all roles
            self.run_evolution = self._run_agent_evolution
        else:
            self.run_evolution = self._run_roles_evolution

        '''Assuming Learner has 1 Agent per Role'''
        self.num_agents = len(self.roles)
        self.observation_space = self.env_specs['observation_space']
        self.action_space = self.env_specs['action_space']
        self.log("Obs space {} / Action space {}".format(self.observation_space, self.action_space), verbose_level=2)

        # Initialize inter components
        self.alg = self.create_algorithm()
        self.buffer = self.create_buffer()
        self.agents = self.create_agents()

        # Check in with Meta
        self.meta.gather(self._get_learner_specs(), root=0)

        # make first saving
        Admin.checkpoint(self, checkpoint_num=0, function_only=True, use_temp_folder=True)

        # Connect with MultiEnvs
        self._connect_menvs()
        self.log("Running..", verbose_level=1)
        self.run()

    def run(self):
        self.step_count = {agent.id:0 for agent in self.agents}
        self.done_count = 0
        self.num_updates = 0
        self.steps_per_episode = {agent.id:0 for agent in self.agents}
        self.metrics_env = {agent.id:[] for agent in self.agents}
        self.last_rewards = {agent.id:{'q':deque(maxlen=self.configs['Agent']['lr_decay']['average_episodes']), 'n':0} for agent in self.agents}

        self.profiler.start(["AlgUpdates", 'ExperienceReceived'])
        while self.done_count < self.episodes:
            self.check_incoming_trajectories()
            self.run_updates()
            self.run_evolution()
            self.collect_metrics(episodic=True) # tensorboard

        self.close()

    def check_incoming_trajectories(self):
        self.last_metric_received = None

        self._n_success_pulls = 0
        self.metrics_env = {agent.id:[] for agent in self.agents}
        for _ in range(self.n_traj_pulls):
            for comm in self.envs:
                self.receive_trajectory_numpy(comm)
        if self._n_success_pulls > 0:
            self.profiler.time('ExperienceReceived', self.done_count, output_quantity=self._n_success_pulls)
        if self.last_metric_received is not None: # and self.done_count % 20 == 0:
            self.log("{} {}:{}".format(self._n_success_pulls, self.done_count, self.last_metric_received), verbose_level=1)

    def receive_trajectory_numpy(self, env_comm):
        '''Receive trajectory from each single environment in self.envs process group'''
        # env_comm = self.envs
        if env_comm.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=self.info):
            env_source = self.info.Get_source()
            self.traj_info = env_comm.recv(None, source=env_source, tag=Tags.trajectory_info) # block statement

            traj_length_index = self.traj_info['length_index']
            traj_length = self.traj_info['obs_shape'][traj_length_index]
            # assert role == self.roles, "<Learner{}> Got trajectory for {} while we expect for {}".format(self.id, role, self.roles)
            assert set(self.roles).issuperset(set(self.traj_info['role'])), "<Learner{}> Got trajectory for {} while we expect for {}".format(self.id, self.traj_info['role'], self.roles)

            observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
            env_comm.Recv([observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_observations)

            actions = np.empty(self.traj_info['acs_shape'], dtype=np.float64)
            env_comm.Recv([actions, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_actions)

            rewards = np.empty(self.traj_info['rew_shape'], dtype=np.float64)
            env_comm.Recv([rewards, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_rewards)

            next_observations = np.empty(self.traj_info['obs_shape'], dtype=np.float64)
            env_comm.Recv([next_observations, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_next_observations)

            dones = np.empty(self.traj_info['done_shape'], dtype=np.float64)
            env_comm.Recv([dones, MPI.DOUBLE], source=env_source, tag=Tags.trajectory_dones)

            # self.step_count += traj_length
            self.done_count += 1
            # self.steps_per_episode = traj_length
            # below metric could be a potential issue when Learner controls multiple agents and receives individual trajectories for them
            self.reward_per_episode = sum(rewards.squeeze())
            self._n_success_pulls += 1
            self.log("Got TrajectoryInfo\n{}".format(self.traj_info), verbose_level=3)

            for role_ix, role in enumerate(self.traj_info['role']):
                '''Assuming 1 agent per role here'''
                agent_id = self.role2ids[role][0]
                self.step_count[agent_id] += traj_length
                self.steps_per_episode[agent_id] = traj_length
                for metric_ix, metric_set in enumerate(self.traj_info['metrics'][role_ix]):
                    self.traj_info['metrics'][role_ix][metric_ix] += (self.done_count,) # add the x_value for tensorboard!
                self.metrics_env[agent_id] += self.traj_info['metrics'][role_ix]
                self.last_rewards[agent_id]['q'].append(self.reward_per_episode)

            self.last_metric_received = f"{self.traj_info['env_id']} got ObsShape {observations.shape} {self.traj_info['metrics']}"

            # self.log(f"Obs {observations.shape} {observations}")
            # self.log(f"Acs {actions}")
            # self.log(f"Rew {rewards.shape} {rewards}")
            # self.log(f"NextObs {next_observations}")
            # self.log(f"Dones {dones}")

            '''VERY IMPORTANT Assuming each individual role has same acs/obs dimension and reward function'''
            exp = list(map(torch.clone, (torch.from_numpy(observations).reshape(traj_length, len(self.traj_info['role']), observations.shape[-1]),
                                         torch.from_numpy(actions).reshape(traj_length, len(self.traj_info['role']), actions.shape[-1]),
                                         torch.from_numpy(rewards).reshape(traj_length, len(self.traj_info['role']), rewards.shape[-1]),
                                         torch.from_numpy(next_observations).reshape(traj_length, len(self.traj_info['role']), next_observations.shape[-1]),
                                         torch.from_numpy(dones).reshape(traj_length, len(self.traj_info['role']), dones.shape[-1])
                                         )))
            self.buffer.push(exp)

    def run_updates(self):
        '''Training'''
        if not self.evaluate \
                and (self.done_count % self.episodes_to_update == 0) \
                and len(self.buffer) > self.buffer.batch_size:

            self.alg.update(self.agents, self.buffer, self.done_count, episodic=True)
            self.num_updates = self.alg.get_num_updates()
            self.profiler.time('AlgUpdates', self.num_updates, output_quantity=self.alg.update_iterations)

            _decay_or_restore_lr = 0 # -1 to decay, 1 to restore, 0 to do nothing
            _decay_log = ""
            for agent in self.agents:
                agent.step_count = self.step_count[agent.id]
                agent.done_count = self.done_count
                agent.num_updates = self.num_updates
                # update this HPs so that they show up on tensorboard
                agent.recalculate_hyperparameters()
                if 'lr_decay' in self.configs['Agent'] and self.configs['Agent']:
                    if not agent.is_exploring() and self.done_count > (self.configs['Agent']['lr_decay']['wait_episodes_to_decay'] * self.last_rewards[agent.id]['n']):
                        self.last_rewards[agent.id]['n'] = self.done_count // self.configs['Agent']['lr_decay']['wait_episodes_to_decay'] + 1
                        agent_ave_reward = sum(self.last_rewards[agent.id]['q']) / len(self.last_rewards[agent.id]['q'])
                        if self.configs['Environment']['expert_reward_range'][agent.role][0] <= agent_ave_reward <= self.configs['Environment']['expert_reward_range'][agent.role][1]:
                            agent.decay_learning_rate()
                            _decay_or_restore_lr = -1
                            _decay_log += f"Decay Actor LR {agent.actor_learning_rate}"
                        else:
                            agent.restore_learning_rate()
                            _decay_or_restore_lr = 1
                            _decay_log += f"Restore Actor LR {agent.actor_learning_rate}"

            self.alg.decay_learning_rate() if _decay_or_restore_lr == -1 else self.alg.restore_learning_rate() if _decay_or_restore_lr == 1 else None
            self.log(f"{_decay_log} / Critic LR {self.alg.critic_learning_rate} / Last{self.configs['Agent']['lr_decay']['average_episodes']}AveRew {agent_ave_reward}", verbose_level=1) if _decay_log != '' else None

            '''Save latest updated agent in temp folder for MultiEnv and Evals to load'''
            self.checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=True)

            '''Check point purposes only'''
            if self.done_count % self.save_checkpoint_episodes == 0:
                self.checkpoint(checkpoint_num=self.done_count, function_only=True, use_temp_folder=False)

    def _run_roles_evolution(self):
        '''Roles Evolution'''

        '''
            Expectation value of how many parameter to change per evolution = 1
        '''
        if self.pbt and self.done_count >= self.initial_evolution_episodes:

            if (self.done_count - self.initial_evolution_episodes) >= (self.n_evolution_requests * self.evolution_episodes):
                self.meta.send(self._get_learner_specs(), dest=0, tag=Tags.evolution_request) # ask for evolution configs1
                self.n_evolution_requests += 1
                # self.log("Ask for Evolution", verbose_level=3)

            if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.evolution_config, status=self.info):
                self.evolution_config = self.meta.recv(None, source=self.info.Get_source(), tag=Tags.evolution_config)  # block statement

                if not self.evolve:
                    self.log("self.evolve is set to {}! Got Evolution {}".format(self.evolve, self.evolution_config), verbose_level=1)
                    return

                self.log('Got Evolution {}'.format(self.evolution_config), verbose_level=1)
                self.evolution_count += 1

                for evol_config in self.evolution_config:
                    if evol_config['evolution'] == False or evol_config['agent_id'] == evol_config['evo_agent_id']:
                        continue
                    agent = self.get_agent_of_id(evol_config['agent_id'])
                    setattr(self, 'exploitation', getattr(self, evol_config['exploitation']))
                    setattr(self, 'exploration', getattr(self, evol_config['exploration']))
                    self.exploitation(agent, evol_config)
                    self.exploration(agent)

                '''No need to send message to MultiEnv for now'''
                # for ix in range(self.num_menvs):
                #     self.menv.send(self._get_learner_state(), dest=ix, tag=Tags.new_agents)

                # self.log('Evolution Completed for {} agents'.format(len(self.agents)), verbose_level=2)
            # else:
            #     self.log("No configs probed!", verbose_level=1)

    def create_agents(self):
        assert hasattr(self, 'num_agents') and self.num_agents > 0, 'Learner num_agent not specified, got {}'.format(self.num_agents)
        self.start_agent_idx = (self.id+1) * 1000
        self.new_agents_ids = np.arange(self.start_agent_idx, self.start_agent_idx + self.num_agents)

        if self.load_agents:
            agents = Admin._load_agents(self.load_agents, absolute_path=False, load_latest=False, device=self.alg.device) # the alg determines the device
            # minor tweak on the agent id as we can have an issue when multiple learners load the same agent id (like loading a PBT session)
            for a in agents:
                a.id += 10 * self.id
            self.alg.add_agents(agents)
            agent_creation_log = "{} agents loaded".format([str(a) for a in agents])
        elif hasattr(self, 'roles') and len(self.roles) > 0:
            self.agents_dict = {role:self.alg.create_agent_of_role(self.new_agents_ids[ix], role) for ix, role in enumerate(self.roles)}
            agents = list(self.agents_dict.values())
            agent_creation_log = "{} agents created: {}".format(len(agents), [str(a) for a in agents])
        else:
            agents = [self.alg.create_agent(ix) for ix in self.new_agents_ids]
            agent_creation_log = "{} agents created: {}".format(len(agents), [str(a) for a in agents])

        self.log(agent_creation_log, verbose_level=1)

        self.role2ids = {role:[] for role in self.roles}
        self.id2role = {}
        self.metrics_env = {}
        for _agent in agents:
            _agent.to_device(self.alg.device)
            _agent.evaluate = self.evaluate
            self.role2ids[_agent.role] += [_agent.id]
            self.metrics_env[_agent.id] = []
            self.id2role[_agent.id] = _agent.role
            _agent.recalculate_hyperparameters()
        return agents

    def get_agent_of_id(self, id):
        for agent in self.agents:
            if agent.id == id:
                return agent

    def create_algorithm(self):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        self.configs['Algorithm']['roles'] = self.roles if hasattr(self, 'roles') else []
        alg = algorithm_class(self.observation_space, self.action_space, self.configs)
        self.log("Algorithm created of type {}".format(algorithm_class), verbose_level=2)
        return alg

    def create_buffer(self):
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        if type(self.observation_space) == dict:
            '''Assuming roles with same obs/acs dim'''
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'],
                                  self.num_agents, self.observation_space[self.roles[0]],
                                  sum(self.action_space[self.roles[0]]['acs_space']))
        else:
            buffer = buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.observation_space, self.action_space['acs_space'])
        self.log("Buffer created of type {}".format(buffer_class), verbose_level=2)
        return buffer

    def _connect_io_handler(self):
        self.io = get_io_stub(self.configs)

    def checkpoint(self, checkpoint_num, function_only, use_temp_folder):
        for a in self.agents:
            self.alg.save_central_critic(a)

        if use_temp_folder:
            self.io.request_io(self._get_learner_specs(), Admin.get_learner_url(self), wait_for_access=True)
        Admin.checkpoint(self, checkpoint_num=checkpoint_num, function_only=function_only, use_temp_folder=use_temp_folder)
        if use_temp_folder:
            self.io.done_io(self._get_learner_specs(), Admin.get_learner_url(self))

    def t_test(self, agent, evo_config):
        if evo_config['ranking'] > evo_config['evo_ranking']:

            self.io.request_io(self._get_learner_specs(), self.eval_path, wait_for_access=True)
            path = self.eval_path + 'Agent_' + str(evo_config['agent_id'])
            evo_path = self.eval_path + 'Agent_'+str(evo_config['evo_agent_id'])
            if 'RoboCup' in self.configs['Environment']['type']:
                with open(self.eval_path+'Agent_'+str(evo_config['agent_id'])+'/episode_evaluations.data','rb') as file_handler:
                    evals = np.array(pickle.load(file_handler))
                with open(self.eval_path+'Agent_'+str(evo_config['evo_agent_id'])+'/episode_evaluations.data','rb') as file_handler:
                    evo_evals = np.array(pickle.load(file_handler))
            else:
                with open(path + '_episode_evaluations.data', 'rb') as file_handler:
                    evals = np.array(pickle.load(file_handler))
                with open(evo_path + '_episode_evaluations.data', 'rb') as file_handler:
                    evo_evals = np.array(pickle.load(file_handler))
            self.io.done_io(self._get_learner_specs(), self.eval_path)

            if self.welch_T_Test(evals, evo_evals):
                self.truncation(agent, evo_config)

    def welch_T_Test(self, evals, evo_evals):
        if 'RoboCup' in self.configs['Environment']['type']:
            return True
        else:
            t, p = stats.ttest_ind(evals, evo_evals, equal_var=False)
            return p < self.p_value

    def truncation(self, agent, evo_config):
        self.io.request_io(self._get_learner_specs(), evo_config['evo_path'], wait_for_access=True)
        evo_agent = Admin._load_agent_of_id(evo_config['evo_path'], evo_config['evo_agent_id'])[0]
        self.io.done_io(self._get_learner_specs(), evo_config['evo_path'])

        agent.copy_hyperparameters(evo_agent)
        self.alg.copy_hyperparameters(evo_agent)

        agent.copy_weights(evo_agent)
        self.alg.copy_weight_from_agent(evo_agent)

    def perturb(self, agent):
        perturb_factor = np.random.choice(self.perturb_factor)
        agent.perturb_hyperparameters(perturb_factor)
        self.alg.perturb_hyperparameters(perturb_factor)

    def resample(self, agent):
        agent.resample_hyperparameters()
        self.alg.resample_hyperparameters()

    def exploitation(self):
        raise NotImplemented

    def exploration(self):
        raise NotImplemented

    def _connect_menvs(self):
        # Connect with MultiEnv
        self.menv = MPI.COMM_WORLD.Connect(self.menv_port,  MPI.INFO_NULL)
        self.log('Connected with MultiEnvs', verbose_level=2)

        for i in range(self.num_menvs):
            self.menv.send(self._get_learner_specs(), dest=i, tag=Tags.specs)

        # We need to accept the connection from each Environments group == number of MultiEnvs
        self.envs = []
        for env_port in self.port['env']:
            self.envs.append(MPI.COMM_WORLD.Accept(env_port))
        # self.envs = MPI.COMM_WORLD.Accept(self.port)
        self.log("Connected with {} Env Groups".format(len(self.envs)), verbose_level=2)

    def _get_learner_state(self):
        specs = self._get_learner_specs()
        specs['num_updates'] = self.num_update
        return specs

    def _get_learner_specs(self):
        return {
            'type': 'Learner',
            'id': self.id,
            'evaluate': self.evaluate,
            'roles': self.roles,
            'num_agents': self.num_agents,
            'agent_ids': list([a.id for a in self.agents]) if hasattr(self, 'agents') else None,
            'role2ids': self.role2ids,
            'done_count': self.done_count if hasattr(self, 'done_count') else None,
            'port': self.port,
            'menv_port': self.menv_port,
            'load_path': Admin.get_learner_url(self),
        }

    def get_metrics(self, episodic, agent_id):
        evolution_metrics = []
        # if self.pbt:
        agent = self.get_agent_of_id(agent_id)
        evolution_metrics += agent.get_metrics()
        # evolution_metrics += [('Agent/{}/Actor_Learning_Rate'.format(agent.role), agent.actor_learning_rate)]
        return [self.metrics_env[agent_id]] + evolution_metrics

    def set_default_configs(self):
        assert 'Learner' in self.configs, 'No Learner config found on received config: {}'.format(self.configs)
        default_configs = {
            'evaluate': False,
            'n_traj_pulls': 1,
            'episodes': float('inf')
            # 'perturb_factor': [0.8, 1.2]
            # add default configs here

        }
        for attr_name, default_val in default_configs.items():
            if attr_name not in self.configs['Learner']:
                self.configs['Learner'][attr_name] = default_val

        assert 'Agent' in self.configs, "No Agent config given!, got {}".format(self.configs)
        default_agent_configs = {
            'lr_decay': {'factor': 1, 'average_episodes': 100, 'wait_episodes_to_decay': 100} # no decay by default
        }
        for attr_name, default_val in default_agent_configs.items():
            if attr_name not in self.configs['Agent']:
                self.configs['Agent'][attr_name] = default_val

    def close(self):
        self.log("Started closing", verbose_level=2)
        self.meta.send(self._get_learner_specs(), dest=0, tag=Tags.close)
        for e in self.envs:
            e.Disconnect()
        # Close Environment port
        for portname in self.port['env']:
            MPI.Close_port(portname)
        self.log("Closed Environments", verbose_level=2)
        self.menv.Disconnect()
        self.log("Closed MultiEnv", verbose_level=2)
        self.meta.Disconnect()
        self.log("FULLY CLOSED", verbose_level=1)
        exit(0)

    def __str__(self):
        return "<Learner(id={})>".format(self.id)

if __name__ == "__main__":
    try:
        l = MPILearner()
    except Exception as e:
        msg = "<Learner(id={})> error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
        terminate_process()
    finally:
        pass
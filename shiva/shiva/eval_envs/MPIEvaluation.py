import sys, time, traceback, pickle,torch
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import numpy as np

from shiva.eval_envs.Evaluation import Evaluation
from shiva.helpers.misc import terminate_process, flat_1d_list
from shiva.helpers.utils.Tags import Tags
from shiva.core.admin import Admin, logger
from shiva.core.IOHandler import get_io_stub


class MPIEvaluation(Evaluation):
    """ Manages Instances of MPIEvalEnv, managed by MPIMultiEvalWrapper
    Hosts an agent to give actions for all the environments.
    """
    def __init__(self):
        self.meval = MPI.Comm.Get_parent()
        self.id = self.meval.Get_rank()
        self.info = MPI.Status()
        self.launch()

    def launch(self):
        """ Launches the Evaluation Instance

        Gets the configs from multienvironment, connects to the IO Handler, launches the environments,

        """
        # Receive Config from MultiEvalWrapper
        self.configs = self.meval.bcast(None, root=0)
        super(MPIEvaluation, self).__init__(self.configs)
        Admin.init(self.configs)

        self._connect_io_handler()

        if hasattr(self, 'device') and self.device == 'gpu':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')

        self._launch_envs()
        self.meval.gather(self._get_eval_specs(), root=0) # checkin with MultiEvalWrapper

        if 'Gym' in self.configs['Environment']['type'] \
                or 'Unity' in self.configs['Environment']['type'] \
                or 'ParticleEnv' in self.configs['Environment']['type']:
            self._receive_new_match() # receive first match
            self._receive_eval = self._receive_roles_evals
            self.done_evaluating = self.done_evaluating_roles
            self.send_eval_update_agents = self.send_roles_eval_update_agents
        else:
            assert False, "Environment type not able to evaluate"

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)
        self.log('Eval Envs have been told to start!', verbose_level=1)
        self.run()

    def run(self):
        self.step_count = 0

        if 'Unity' in self.env_specs['type'] or 'ParticleEnv' in self.env_specs['type']:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0] ), dtype=np.float64)
        elif 'Gym' in self.env_specs['type']:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space'] ), dtype=np.float64)
        elif 'RoboCup' in self.env_specs['type']:
            self._obs_recv_buffer = np.empty((self.num_envs, self.env_specs['num_agents'], self.env_specs['observation_space']), dtype=np.float64)


        while True:
            time.sleep(self.configs['Admin']['time_sleep']['Evaluation'])
            self._receive_eval()
            self._step_python()
            # self._step_numpy()
            if self.done_evaluating():
                self.send_eval_update_agents()

    def _step_python(self):
        self._obs_recv_buffer = self.envs.gather(None, root=MPI.ROOT)
        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if 'Unity' in self.env_specs['type']:
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    # role_actions = []
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    # try batching all role observations to the agent
                    # role_actions.append(self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate']))
                    # for o in role_obs:
                    #     role_actions.append(self.agents[agent_ix].get_action(o, self.step_count, evaluate=self.role2learner_spec[role_name]['evaluate']))
                    # env_actions.append(role_actions)
                    env_actions.append(self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=not self.allow_noise))
                actions.append(env_actions)
        elif 'Particle' in self.env_specs['type']:
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=not self.allow_noise)
                    env_actions.append(role_actions)
                actions.append(env_actions)
        elif 'Gym' in self.env_specs['type']:
            actions = []
            role_ix = 0
            role_name = self.env_specs['roles'][role_ix]
            agent_ix = self.role2agent[role_name]
            for role_obs in self._obs_recv_buffer:
                env_actions = []
                role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=not self.allow_noise)
                env_actions.append(role_actions)
                actions.append(env_actions)
        self.actions = np.array(actions)
        self.log("Step {} Obs {} Acs {}".format(self.step_count, self._obs_recv_buffer, actions), verbose_level=3)
        self.envs.scatter(actions, root=MPI.ROOT)

    def _step_numpy(self):
        self.envs.Gather(None, [self._obs_recv_buffer, MPI.DOUBLE], root=MPI.ROOT)
        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if 'Unity' in self.env_specs['type']:
            actions = [[[self.agents[ix].get_action(o, self.step_count, evaluate=True) for o in obs] for ix, obs in
                        enumerate(env_observations)] for env_observations in self._obs_recv_buffer]
            self.actions = np.array(actions)
            self.envs.scatter(self.actions, root=MPI.ROOT)
        elif 'Gym' in self.env_specs['type']:
            actions = [[agent.get_action(obs, self.step_count, evaluate=True) for agent, obs in zip(self.agents, observations)]
                       for observations in self._obs_recv_buffer]
            self.actions = np.array(actions)
            self.envs.scatter(self.actions, root=MPI.ROOT)
        elif 'RoboCup' in self.env_specs['type']:
            actions = [[agent.get_action(obs, self.step_count, evaluate=True) for agent, obs in zip(self.agents, observations)]
                       for observations in self._obs_recv_buffer]
            actions = np.array(actions, dtype=np.float64)
            self.envs.Scatter([actions, MPI.DOUBLE], None, root=MPI.ROOT)

    '''
        Roles Methods
    '''

    def done_evaluating_roles(self):
        return len(self.eval_metrics) >= self.eval_episodes

    def _receive_roles_evals(self):
        '''Receive metrics after every episode from each Environment'''
        while self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_eval, status=self.info):
            env_source = self.info.Get_source()
            env_metrics = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)
            self.eval_metrics.append(env_metrics)
            self.log("Got Metrics {}".format(env_metrics), verbose_level=2)

    def send_roles_eval_update_agents(self):
        '''Do averaging of metrics across all metrics received, then send'''
        evals = {role:{} for role in self.roles}
        metrics_received = []

        # rename metric
        metric_conversion = {
            'reward_per_episode': 'Average_Reward'
        }

        for episode_metric in self.eval_metrics:
            for metric_name, role_values in episode_metric.items():
                if metric_name not in metrics_received:
                    metrics_received.append(metric_name)

                # @metric_name is 'reward_per_episode' only for now
                for role, value in role_values.items():
                    if metric_name not in evals[role]:
                        evals[role][metric_name] = []
                    evals[role][metric_name].append(value)

        for role in self.roles:
            learner_spec = self.role2learner_spec[role]
            file_path = f"{learner_spec['load_path']}/evaluations/"

            for metric_name in metrics_received:
                evals[role][metric_conversion[metric_name]] = np.mean(evals[role][metric_name])

                if metric_name == 'reward_per_episode':
                    self.io.request_io(self._get_eval_specs(), file_path, wait_for_access=True)
                    file_name = f"{file_path}/Agent_{str(self.agents[self.role2agent[role]].id)}.npy"
                    np.save(file_name, np.array(evals[role][metric_name]))
                    self.io.done_io(self._get_eval_specs(), file_path)

                    self.log("Saved Evals {} @ {}".format(evals[role][metric_name], file_name), verbose_level=3)
                del evals[role][metric_name]

        self.meval.send(evals, dest=0, tag=Tags.evals)
        self.log("Summarized {} into {}".format(self.eval_metrics, evals), verbose_level=2)
        self._receive_new_match()

    def _receive_new_match(self):
        '''Is a blocking function - chances are that there will be a message'''
        # self.role2id = self.meval.recv(None, source=0, tag=Tags.new_agents)
        self.role2learner_spec = self.meval.recv(None, source=0, tag=Tags.new_agents)
        self.log("Received Match {}".format(self.role2learner_spec), verbose_level=3)
        self.agents = self.load_agents(self.role2learner_spec)
        self.agent_ids = [a.id for a in self.agents]
        self.role2agent = self.get_role2agent()
        self.eval_metrics = []
        self.log("Got match for: {}".format(self.agent_ids), verbose_level=2)

    def get_role2agent(self):
        '''Create mapping of Role->Agent_index in self.agents list'''
        self.role2agent = {}
        for role in self.env_specs['roles']:
            for ix, agent in enumerate(self.agents):
                if role == agent.role:
                    self.role2agent[role] = ix
                    break
        return self.role2agent

    def load_agents(self, role2learner_spec=None):
        """

        """
        if role2learner_spec is None:
            role2learner_spec = self.role2learner_spec

        agents = self.agents if hasattr(self, 'agents') else [None for i in range(len(self.env_specs['roles']))]
        for role, learner_spec in role2learner_spec.items():
            '''Need to reload ONLY the agents that are not being evaluated'''
            if not learner_spec['evaluate']:

                # Useful when loading all Learners agents
                # learner_agents = Admin._load_agents(learner_spec['load_path'])
                # for a in learner_agents:
                #     agents[self.env_specs['roles'].index(a.role)] = a

                # Here when loading individual Agents
                '''Assuming Learner has 1 Agent per Role'''
                agent_id = learner_spec['role2ids'][role][0]

                self.io.request_io(self._get_eval_specs(), learner_spec['load_path'], wait_for_access=True)
                agent = Admin._load_agent_of_id(learner_spec['load_path'], agent_id, device=self.device)[0]
                self.io.done_io(self._get_eval_specs(), learner_spec['load_path'])

                agent.to_device(self.device)
                agents[self.env_specs['roles'].index(agent.role)] = agent

        self.log("Loaded {}".format([str(agent) for agent in agents]), verbose_level=1)
        return agents

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvalEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces)
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them

    def _connect_io_handler(self):
        self.io = get_io_stub(self.configs)

    def _get_eval_specs(self):
        return {
            'type': 'Evaluation',
            'id': self.id,
            'env_specs': self.env_specs,
            'num_envs': self.num_envs
        }

    def close(self) -> None:
        """ Close connection with MultiEvaluationWrapper

        Returns:
            None
        """
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def __str__(self):
        return "<Eval(id={}, device={})>".format(self.id, self.device)

if __name__ == "__main__":
    try:
        eval = MPIEvaluation()
    except Exception as e:
        msg = "<Eval(id={})> error: {}".format(MPI.Comm.Get_parent().Get_rank(), traceback.format_exc())
        print(msg)
        logger.info(msg, True)
    finally:
        terminate_process()

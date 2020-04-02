import sys, time, traceback, pickle,torch
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from mpi4py import MPI
import numpy as np

from shiva.eval_envs.Evaluation import Evaluation
from shiva.helpers.misc import terminate_process, flat_1d_list
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin

class MPIEvaluation(Evaluation):

    def __init__(self):
        self.meval = MPI.Comm.Get_parent()
        self.id = self.meval.Get_rank()
        self.info = MPI.Status()
        self.launch()

    def launch(self):
        # Receive Config from MultiEvalWrapper
        self.configs = self.meval.bcast(None, root=0)
        super(MPIEvaluation, self).__init__(self.configs)
        Admin.init(self.configs)

        self._connect_io_handler()

        # self.device = torch.device('cpu')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._launch_envs()
        self.meval.gather(self._get_eval_specs(), root=0) # checkin with MultiEvalWrapper

        '''Set functions and data structures'''
        if 'RoboCup' in self.env_specs['type']:
            self.agent_sel = self.meval.recv(None, source=0, tag=Tags.new_agents)
            self._io_load_agents()
            self.log("Got Agents {}".format([str(a) for a in self.agents]), verbose_level=1)
            self.agent_ids = [id for id in self.agent_sel]
            print('Agent IDs: ', self.agent_ids)
            print('Agent Sel: ', self.agent_sel)
            self.evals_list = [[None] * self.eval_episodes] * self.agents_per_env
            self.send_eval_update_agents = getattr(self, 'send_robocup_eval_update_agents')
            self._receive_eval = self._receive_eval_numpy
            self.ep_evals = dict()
            self.eval_counts = np.zeros(len(self.agent_ids), dtype=int)
        elif 'Gym' in self.configs['Environment']['type'] \
                or 'Unity' in self.configs['Environment']['type'] \
                or 'ParticleEnv' in self.configs['Environment']['type']:
            self._receive_new_match() # receive first match
            self._receive_eval = self._receive_roles_evals
            self.done_evaluating = self.done_evaluating_roles
            self.send_eval_update_agents = self.send_roles_eval_update_agents
            self.eval_metrics = []
            self.eval_counts = 0
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
                # for i in range(self.num_envs):
                #     self.envs.send([True], dest=i, tag=Tags.clear_buffers)
                # print("Agents have been told to clear buffers for new agents")


    def _step_python(self):
        self._obs_recv_buffer = self.envs.gather(None, root=MPI.ROOT)
        self.step_count += self.env_specs['num_instances_per_env'] * self.num_envs
        if 'Unity' in self.env_specs['type']:
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_actions = []
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    for o in role_obs:
                        role_actions.append(self.agents[agent_ix].get_action(o, self.step_count, evaluate=True))
                    env_actions.append(role_actions)
                actions.append(env_actions)
        elif 'Particle' in self.env_specs['type']:
            actions = []
            for env_observations in self._obs_recv_buffer:
                env_actions = []
                for role_ix, role_name in enumerate(self.env_specs['roles']):
                    role_obs = env_observations[role_ix]
                    agent_ix = self.role2agent[role_name]
                    role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=True)
                    env_actions.append(role_actions)
                actions.append(env_actions)
        elif 'Gym' in self.env_specs['type']:
            actions = []
            role_ix = 0
            role_name = self.env_specs['roles'][role_ix]
            agent_ix = self.role2agent[role_name]
            for role_obs in self._obs_recv_buffer:
                env_actions = []
                role_actions = self.agents[agent_ix].get_action(role_obs, self.step_count, evaluate=True)
                env_actions.append(role_actions)
                actions.append(env_actions)
        self.actions = np.array(actions)
        self.log("Step {} Obs {} Acs {}".format(self.step_count, self._obs_recv_buffer, actions), verbose_level=2)
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
        if self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_eval, status=self.info):
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
        self.log("To summarize {}".format(self.eval_metrics), verbose_level=2)

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
            for metric_name in metrics_received:
                evals[role][metric_conversion[metric_name]] = np.mean(evals[role][metric_name])
                del evals[role][metric_name]

        self.meval.send(evals, dest=0, tag=Tags.evals)
        self.log("Sent Tags.evals {}".format(evals), verbose_level=2)
        self._receive_new_match()

    def _receive_new_match(self):
        '''Is a blocking function - chances are that there will be a message'''
        # self.role2id = self.meval.recv(None, source=0, tag=Tags.new_agents)
        self.role2learner_spec = self.meval.recv(None, source=0, tag=Tags.new_agents)
        self.log("Received Match {}".format(self.role2learner_spec), verbose_level=3)
        '''Assuming a Learner has 1 Agent per Role'''
        self.agent_ids = [self.role2learner_spec[role]['role2ids'][role][0] for role in self.roles] # keep same order of the self.roles list
        self.agents = self.load_agents(self.role2learner_spec)
        self.role2agent = self.get_role2agent()
        self.log("Got match for: {}".format(self.agent_ids), verbose_level=2)

    def get_role2agent(self):
        '''Create Role->AgentIX mapping'''
        self.role2agent = {}
        for role in self.env_specs['roles']:
            for ix, agent in enumerate(self.agents):
                if role == agent.role:
                    self.role2agent[role] = ix
                    break
        return self.role2agent

    def load_agents(self, role2learner_spec=None):
        if role2learner_spec is None:
            role2learner_spec = self.role2learner_spec
        self.io.send(True, dest=0, tag=Tags.io_eval_request)
        _ = self.io.recv(None, source=0, tag=Tags.io_eval_request)
        agents = self.agents if hasattr(self, 'agents') else [None for i in range(len(self.env_specs['roles']))]

        for role, learner_spec in role2learner_spec.items():
            '''Need to load ONLY the agents that are not being evaluated'''
            if not learner_spec['evaluate']:

                # Useful when loading all Learners agents
                # learner_agents = Admin._load_agents(learner_spec['load_path'])
                # for a in learner_agents:
                #     agents[self.env_specs['roles'].index(a.role)] = a

                # Here when loading individual Agents
                '''Assuming Learner has 1 Agent per Role'''
                agent_id = learner_spec['role2ids'][role][0]
                agent = Admin._load_agent_of_id(learner_spec['load_path'], agent_id)
                agent.to_device(self.device)
                agents[self.env_specs['roles'].index(agent.role)] = agent

        self.io.send(True, dest=0, tag=Tags.io_eval_request)
        self.log("Loaded {}".format([str(agent) for agent in agents]), verbose_level=1)
        return agents

    '''
        Single Agent Methods
    '''

    def done_evaluating(self):
        return self.eval_counts.sum() >= self.eval_episodes * self.agents_per_env

    def send_eval_update_agents(self):
        if self.eval_counts.sum() >= self.eval_episodes*self.agents_per_env:
            print('Sending Eval and updating most recent agent file path ')
            for i in range(self.agents_per_env):
                self.log('agent_id: {}'.format(self.agent_ids[i]))
                path = self.eval_path+'Agent_'+str(self.agent_ids[i])
                self.log('Sending Evaluations to MultiEval: {}'.format(self.evals[i]))
                self.meval.send(self.agent_ids[i],dest=0,tag=Tags.evals)
                self.meval.send(self.evals[i],dest=0,tag=Tags.evals)
                #self.ep_evals['path'] = path+'/episode_evaluations'
                #self.ep_evals['evals'] = self.evals[i]
                self.io.send(True, dest=0, tag=Tags.io_eval_request)
                _ = self.io.recv(None, source = 0, tag=Tags.io_eval_request)
                np.save(path+'/episode_evaluations',self.evals[i])
                # self.agents = Admin._load_agents(self.eval_path+'Agent_'+str(self.id))
                new_agent = self.meval.recv(None,source=0,tag=Tags.new_agents)[0][0]
                self.agent_ids[i] = new_agent
                print('New Eval Agent: {}'.format(new_agent))
                path = self.eval_path+'Agent_'+str(new_agent)
                self.log('Path: {} '.format(path))
                self.agents[i] = Admin._load_agents(path)[0]
                self.log('Agent: {}'.format(str(self.agents[0])))
                self.evals[i].fill(0)
                self.eval_counts[i]=0
                self.io.send(True, dest=0, tag=Tags.io_eval_request)
            time.sleep(0.1)

    def send_robocup_eval_update_agents(self):
        if self.eval_counts.sum() >= self.eval_episodes*self.agents_per_env:
            print('Sending Eval and updating most recent agent file path ')
            for i in range(self.agents_per_env):
                self.log('agent_id: {}'.format(self.agent_ids[i]))
                path = self.eval_path+'Agent_'+str(self.agent_ids[i])
                self.log('Sending Evaluations to MultiEval: {}'.format(self.evals_list[i]))
                self.meval.send(self.agent_ids[i],dest=0,tag=Tags.evals)
                self.meval.send(self.evals_list[i], dest=0, tag=Tags.evals)
                #self.ep_evals['path'] = path+'/episode_evaluations'
                #self.ep_evals['evals'] = self.evals[i]
                self.io.send(True, dest=0, tag=Tags.io_eval_request)
                _ = self.io.recv(None, source = 0, tag=Tags.io_eval_request)
                #np.save(path+'/episode_evaluations',self.evals[i])
                with open(path+'/episode_evaluations.data','wb') as file_handler:
                    pickle.dump(self.evals_list,file_handler)
                # self.agents = Admin._load_agents(self.eval_path+'Agent_'+str(self.id))
                new_agent = self.meval.recv(None,source=0,tag=Tags.new_agents)[0][0]
                self.agent_ids[i] = new_agent
                print('New Eval Agent: {}'.format(new_agent))
                path = self.eval_path+'Agent_'+str(new_agent)
                self.log('Path: {} '.format(path))
                self.agents[i] = Admin._load_agents(path)[0]
                self.log('Agent: {}'.format(str(self.agents[0])))
                self.evals_list = [[None]*self.eval_episodes]*len(self.agent_ids)
                self.eval_counts[i]=0
                self.io.send(True, dest=0, tag=Tags.io_eval_request)
            time.sleep(0.1)

    def _receive_eval_numpy(self):
        '''Receive trajectory reward from each single  evaluation environment in self.envs process group'''
        '''Assuming 1 Agent here, may need to iterate thru all the indexes of the @traj'''
        if self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_eval):
            info = MPI.Status()
            agent_idx = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_eval, status=info)
            env_source = info.Get_source()

            '''
                Ideas to optimize -> needs some messages that are not multidimensional
                    - Concat Observations and Next_Obs into 1 message (the concat won't be multidimensional)
                    - Concat
                    '''
            if self.eval_counts[agent_idx] < self.eval_episodes:

                if 'RoboCup' in self.env_specs['type']:
                    evals = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)
                    #self.log('Agent IDX: {}'.format(agent_idx))
                    #self.log('Eval Counts: {}'.format(self.eval_counts[agent_idx]))
                    self.evals_list[agent_idx][self.eval_counts[agent_idx]] = evals
                    self.eval_counts[agent_idx] += 1
                else:
                    evals = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)
                    self.evals[agent_idx, self.eval_counts[agent_idx]] = evals
                    self.eval_counts[agent_idx] += 1
            else:
                _ = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)

    def _io_load_agents(self):
        #agent_paths = [self.eval_path+'Agent_'+str(agent_id) for agent_id in self.agent_ids]
        self.io.send(True, dest=0, tag=Tags.io_eval_request)
        _ = self.io.recv(None, source = 0, tag=Tags.io_eval_request)
        self.agents = [Admin._load_agents(self.eval_path+'Agent_'+str(agent_id))[0] for agent_id in self.agent_ids]
        self.log('Load {}'.format([str(a) for a in self.agents]), verbose_level=1)
        self.io.send(True, dest=0, tag=Tags.io_eval_request)

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvalEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces)
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them

    def _connect_io_handler(self):
        self.io = MPI.COMM_WORLD.Connect(self.evals_io_port, MPI.INFO_NULL)
        self.log('Connected with IOHandler', verbose_level=2)

    def _get_eval_specs(self):
        return {
            'type': 'Evaluation',
            'id': self.id,
            'env_specs': self.env_specs,
            'num_envs': self.num_envs
        }

    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def __str__(self):
        return "<Eval(id={})>".format(self.id)

    def show_comms(self):
        self.debug("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.debug("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.debug("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    try:
        MPIEvaluation()
    except Exception as e:
        print("Eval error:", traceback.format_exc())
    finally:
        terminate_process()

import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import logging
from mpi4py import MPI
import numpy as np

from shiva.eval_envs.Evaluation import Evaluation
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin
from shiva.eval_envs.Evaluation import Evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shiva")

class MPIEvaluation(Evaluation):

    def __init__(self):
        self.meval = MPI.Comm.Get_parent()
        self.id = self.meval.Get_rank()
        self.launch()

    def launch(self):
        # Receive Config from MultiEvalWrapper
        self.configs = self.meval.bcast(None, root=0)
        super(MPIEvaluation, self).__init__(self.configs)
        #self.log("Received config with {} keys".format(str(len(self.configs.keys()))))

        self._launch_envs()
        self.meval.gather(self._get_eval_specs(), root=0) # checkin with MultiEvalWrapper
        #self._connect_learners()
        self.agent_sel = self.meval.scatter(None, root=0)
        self.agent_ids = [id for id in self.agent_sel]
        print('Agent IDs: ', self.agent_ids)
        self.evals = np.zeros((len(self.agent_ids),self.eval_episodes))
        self.eval_counts = np.zeros(len(self.agent_ids),dtype=int)
        self.agents = [Admin._load_agents(self.eval_path+'Agent_'+str(agent_id))[0] for agent_id in self.agent_ids]
        #self.log("Agents created: {} of type {}".format(len(self.agents), type(self.agents[0])))

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)

        self.run()

    def run(self):
        self.step_count = 0
        info = MPI.Status()


        try:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], list(self.env_specs['observation_space'].values())[0] ), dtype=np.float64)
        except:
            self._obs_recv_buffer = np.empty(( self.num_envs, self.env_specs['num_agents'], self.env_specs['num_instances_per_env'], self.env_specs['observation_space'] ), dtype=np.float64)


        while True:
            self._receive_eval_numpy()
            self.envs.Gather(None, [self._obs_recv_buffer, MPI.DOUBLE], root=MPI.ROOT)
            # self.debug("Obs {}".format(observations))
            self.step_count  += self.env_specs['num_instances_per_env'] * self.num_envs

            if 'Unity' in self.env_specs['type']:
                actions = [ [ [self.agents[ix].get_action(o, self.step_count, False) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]
            else:
                # Gym
                # same?
                actions = [ [ [self.agents[ix].get_action(o, self.step_count, False) for o in obs] for ix, obs in enumerate(env_observations) ] for env_observations in self._obs_recv_buffer]

            self.actions = np.array(actions)
            self.envs.scatter(self.actions, root=MPI.ROOT)


            if self.eval_counts.sum() >= self.eval_episodes*self.num_agents:

                for i in range(self.num_agents):
                    path = self.eval_path+'Agent_'+str(self.agent_ids[i])
                    self.meval.send(self.agent_ids[i],dest=0,tag=Tags.agent_id)
                    self.meval.send(self.evals[i],dest=0,tag=Tags.evals)
                    np.save(path+'/episode_evaluations',self.evals[i])
                #self.agents = Admin._load_agents(self.eval_path+'Agent_'+str(self.id))
                    self.agents[i] = Admin._load_agents(path)[0]
                    self.evals[i].fill(0)
                    self.eval_counts[i]=0
                    self.envs.bcast([True], root=MPI.ROOT)
                    print("Agents have been told to clear buffers for new agents")
                else:
                    self.envs.bcast([False], root=MPI.ROOT)


        self.close()

    def _launch_envs(self):
        # Spawn Single Environments
        self.envs = MPI.COMM_SELF.Spawn(sys.executable, args=['shiva/eval_envs/MPIEvalEnv.py'], maxprocs=self.num_envs)
        self.envs.bcast(self.configs, root=MPI.ROOT)  # Send them the Config
        envs_spec = self.envs.gather(None, root=MPI.ROOT)  # Wait for Env Specs (obs, acs spaces)
        envs_state = self.envs.gather(None, root=MPI.ROOT)
        assert len(envs_spec) == self.num_envs, "Not all Environments checked in.."
        self.env_specs = envs_spec[0] # set self attr only 1 of them


    def _get_eval_specs(self):
        return {
            'type': 'Evaluation',
            'id': self.id,
            'env_specs': self.env_specs,
            'num_envs': self.num_envs
        }

    def _receive_eval_numpy(self):
        '''Receive trajectory reward from each single  evaluation environment in self.envs process group'''
        '''Assuming 1 Agent here, may need to iterate thru all the indexes of the @traj'''

        if self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_info):
            info = MPI.Status()
            agent_idx = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_info, status=info)
            env_source = info.Get_source()

            '''
                Ideas to optimize -> needs some messages that are not multidimensional
                    - Concat Observations and Next_Obs into 1 message (the concat won't be multidimensional)
                    - Concat
                    '''
            if self.eval_counts[agent_idx] < self.eval_episodes:
                evals = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)
                self.evals[agent_idx,self.eval_counts[agent_idx]] = evals
                print('Eval: ', self.evals[agent_idx,self.eval_counts[agent_idx]])
                self.eval_counts[agent_idx] += 1
            else:
                _ = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)


        # self.debug("{}\n{}\n{}\n{}\n{}".format(type(observations), type(actions), type(rewards), type(next_observations), type(dones)))
        # self.debug("{}\n{}\n{}\n{}\n{}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))
        # self.debug("{}\n{}\n{}\n{}\n{}".format(observations, actions, rewards, next_observations, dones))



    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'Eval Env {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.debug("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.debug("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.debug("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    MPIEvaluation()

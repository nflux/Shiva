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
        self.log("Received config with {} keys".format(len(self.configs.keys())))
        # Open Port for Learners
        #self.port = MPI.Open_port(MPI.INFO_NULL)
        #self.debug("Open port {}".format(self.port))

        # Grab configs
        #self.num_learners = 1 # assuming 1 Learner
        self.num_envs = self.num_instances
        self.evals = np.zeros(self.eval_episodes)
        self.eval_count = 0
        self._launch_envs()
        self.meval.gather(self._get_eval_specs(), root=0) # checkin with MultiEvalWrapper
        #self._connect_learners()
        self.agent_id = self.meval.scatter(None, root=0)
        self.agents = Admin._load_agents(self.eval_path+'Agent_'+str(self.agent_id))
        self.log("Agents created: {} of type {}".format(len(self.agents), type(self.agents[0])))

        # Give start flag!
        self.envs.bcast([True], root=MPI.ROOT)

        self.run()

    def run(self):
        self.step_count = 0
        while True:
            self._receive_eval_numpy()
            observations = self.envs.gather(None, root=MPI.ROOT)
            # self.debug("Obs {}".format(observations))
            self.step_count += len(observations)
            actions = self.agents[0].get_action(observations, self.step_count) # assuming one agent for all obs
            # self.debug("Acs {}".format(actions))
            self.envs.scatter(actions, root=MPI.ROOT)


            if self.eval_count >= self.eval_episodes:
                path = self.eval_path+'Agent_'+str(self.agent_id)
                self.meval.send(self.evals,dest=0,tag=Tags.evals)
                np.save(path+'/episode_evaluations',self.evals)
                #self.agents = Admin._load_agents(self.eval_path+'Agent_'+str(self.id))
                self.agents = Admin._load_agents(path)
                self.evals = np.zeros(self.eval_episodes)
                self.eval_count = 0
                self.envs.bcast([True], root=MPI.ROOT)
                self.log("Agents have been told to clear buffers for new agents")
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
            'num_envs': self.num_instances
        }

    def _receive_eval_numpy(self):
        '''Receive trajectory reward from each single  evaluation environment in self.envs process group'''
        '''Assuming 1 Agent here, may need to iterate thru all the indexes of the @traj'''

        if self.envs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.trajectory_length):
            info = MPI.Status()
            traj_length = self.envs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.trajectory_length, status=info)
            env_source = info.Get_source()

            '''
                Ideas to optimize -> needs some messages that are not multidimensional
                    - Concat Observations and Next_Obs into 1 message (the concat won't be multidimensional)
                    - Concat
                    '''
            self.evals[self.eval_count] = self.envs.recv(None, source=env_source, tag=Tags.trajectory_eval)
            self.eval_count += 1


        # self.debug("{}\n{}\n{}\n{}\n{}".format(type(observations), type(actions), type(rewards), type(next_observations), type(dones)))
        # self.debug("{}\n{}\n{}\n{}\n{}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))
        # self.debug("{}\n{}\n{}\n{}\n{}".format(observations, actions, rewards, next_observations, dones))



    def close(self):
        comm = MPI.Comm.Get_parent()
        comm.Disconnect()

    def log(self, msg, to_print=False):
        text = 'Learner {}/{}\t{}'.format(self.id, MPI.COMM_WORLD.Get_size(), msg)
        logger.info(text, to_print or self.configs['Admin']['print_debug'])

    def show_comms(self):
        self.debug("SELF = Inter: {} / Intra: {}".format(MPI.COMM_SELF.Is_inter(), MPI.COMM_SELF.Is_intra()))
        self.debug("WORLD = Inter: {} / Intra: {}".format(MPI.COMM_WORLD.Is_inter(), MPI.COMM_WORLD.Is_intra()))
        self.debug("META = Inter: {} / Intra: {}".format(MPI.Comm.Get_parent().Is_inter(), MPI.Comm.Get_parent().Is_intra()))


if __name__ == "__main__":
    MPIEvaluation()


import sys, os, argparse, time
from pathlib import Path
from mpi4py import MPI


def create_comm_env(cls, env_id, configs, menv_address, menv_comm):
    '''
        Dynamic Inheritance from @cls
    '''
    class CommEnvironment(cls):
        menv_stub = None # only 1 MultiEnv per individual Env
        learners_stub = {}
        # for centralized learners we could have 2 agents from 2 diff learners
        # it could be solved with a dictionary mapping
        #   agent_id -> learner_client
        trajectories = {}

        def __init__(self, id, configs, menv_address, menv_comm):
            super(CommEnvironment, self).__init__(configs)
            self.id = id
            self.menv_comm = menv_comm # MPI
            self.menv_stub = get_menv_stub(menv_address) # gRPC

            # send my specs to the multi env
            self.menv_stub.send_env_specs(self._get_env_specs())
            # self.debug("Will soon request Learners info")
            time.sleep(3) # give some time here...
            self.debug("Requesting Learners info")
            self.learners_specs = []
            while len(self.learners_specs) == 0:
                self.learners_specs = self.menv_stub.get_learners_specs(self._get_env_specs())
                time.sleep(2)
            self.learners_stub = {}
            for spec in self.learners_specs:
                self.learners_stub[spec['id']] = get_learner_stub(spec['address'])
            # self.debug("Successfully created stubs for the Learner # {} - Waiting ready signal from MultiEnv!!!".format(len(self.learners_specs)))

            self.trajectories = []

            self.menv_comm.bcast(None, root=0) # this is the flag given by the MultiEnv to start!

            self.run()

        def run(self):
            self.debug("Started collection..")
            while True:
                observations = self.get_observations()
                actions = self.menv_stub.get_actions(observations)
                # self.debug(actions)
                next_observations, rewards, dones, _ = self.step(actions)
                #
                #   Collect trajectories in an unstructured buffer of dim
                #   num_agents * timesteps * (obs_dim | acs_dim | next_obs_dim | reward | done)
                #
                # self.debug("{} {} {} {} {}".format(type(observations), type(actions), type(next_observations), type(rewards), type(dones)))
                exp = [list(observations), actions, rewards, list(next_observations), int(dones)]
                self.trajectories.append(exp)

                if self.is_done():
                    self.debug(self.get_metrics(episodic=True))
                    self.reset()
                    '''
                        Need decentralized implementation
                    '''
                    self.learners_stub[0].send_trajectory(self.trajectories)
                    self.trajectories = []


        def _get_env_specs(self):
            return {
                'id': self.id,
                'observation_space': self.get_observation_space(),
                'action_space': self.get_action_space(),
                'num_agents': self.num_agents
            }

        def debug(self, msg):
            print("PID {} Environm\t\t{}".format(os.getpid(), msg))

    return CommEnvironment(env_id, configs, menv_address, menv_comm)

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parent.parent.parent)) # necessary to add Shiva modules to the path

    from shiva.envs.CommMultiEnvironmentServer import get_menv_stub
    from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub
    from shiva.helpers.config_handler import load_class
    parser = argparse.ArgumentParser()
    parser.add_argument("-pa", "--parent-address", required=True, type=str, help='Parent address')
    args = parser.parse_args()

    menv_comm = MPI.Comm.Get_parent()
    size = menv_comm.Get_size()
    rank = menv_comm.Get_rank()

    configs = menv_comm.bcast(None, root=0)
    env_cls = load_class('shiva.envs', configs['Environment']['type'])  # all envs with same config
    create_comm_env(env_cls, rank, configs, args.parent_address, menv_comm)
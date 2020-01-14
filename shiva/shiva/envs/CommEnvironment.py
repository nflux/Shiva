from shiva.helpers.config_handler import load_class

from shiva.envs.CommMultiEnvironmentServer import get_menv_stub
from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub

from shiva.helpers.grpc_utils import (
    from_observations_2_ObservationsProto,
    from_ActionsProto_2_actions,
    from_trajectory_2_TrajectoryProto
)

def create_comm_env(env_id, cls, configs, menv_stub, send_specs_to_menv):
    '''
        Dynamic Inheritance
    '''
    class CommEnvironment(cls):
        menv_stub = None # only 1 MultiEnv per individual Env
        learners_stub = {}
        # for centralized learners we could have 2 agents from 2 diff learners
        # it could be solved with a dictionary mapping
        #   agent_id -> learner_client
        trajectories = {}

        def __init__(self, env_id, configs, menv_address, send_specs_to_menv=False):
            super(CommEnvironment, self).__init__(configs)
            self.id = env_id
            self.menv_stub = get_menv_stub(menv_address)

            env_class = load_class('shiva.envs', self.configs['Environment']['type'])
            self.env = env_class(self.configs)
            self.env.reset()

            if send_specs_to_menv: # a one-time message to send the env specs
                self.menv_stub.send_env_specs(self._get_env_specs())

            learners_info = self.menv_stub.get_learners_info(self.id) # this might take a little to respond as the menv needs a response from meta

            # once we got the learners addresses, create the stubs
            self.learners_stub = {}
            for learner_id, info in learners_info.items():
                self.learners_stub[learner_id] = get_learner_stub(info['address'])

            # we have the learners stub so we are ready to start running
            self.run()

        def run(self):
            while True:
                observations = self.get_observations()
                actions = self.menv_stub.SendObservations(from_observations_2_ObservationsProto(observations))
                next_observations, rewards, dones, _ = self.env.step(from_ActionsProto_2_actions(actions))

                #
                #   Collect trajectories in an unstructured buffer of dim
                #   num_agents * timesteps * (obs_dim | acs_dim | next_obs_dim | reward | done)
                #

                if self.env.is_done():
                    #
                    #   Send the trajectory to each individual learner (specially for decentralized)
                    #
                    for agent_id in self.agent_ids:
                        self.learners_stub[agent_id].SendTrajectory(from_trajectory_2_TrajectoryProto(self.trajectories[agent_id]))

        def _get_env_specs(self):
            return {
                'env_id': self.id,
                'observation_space': self.get_observation_space(),
                'action_space': self.get_action_space(),
                'num_agents': self.num_agents
            }

    return CommEnvironment(env_id, configs, menv_stub, send_specs_to_menv)
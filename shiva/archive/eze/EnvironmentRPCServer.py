import grpc
import multiprocessing
import time, datetime
from concurrent import futures

from shiva.helpers.config_handler import load_class

from shiva.core.communication_objects.env_command_pb2 import EnvironmentCommand
from shiva.core.communication_objects.env_step_pb2 import EnvStepInput, EnvStepOutput
from shiva.core.communication_objects.env_specs_pb2 import EnvironmentSpecs
from shiva.core.communication_objects.service_env_pb2_grpc import EnvironmentServicer, add_EnvironmentServicer_to_server
from shiva.core.communication_objects.helpers_pb2 import Empty

# this might be useful for cloud deployment
_PROCESS_COUNT = multiprocessing.cpu_count()
_THREAD_CONCURRENCY = _PROCESS_COUNT
_ONE_HOUR = datetime.timedelta(hours=1)
#

def serve(address, configs, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_EnvironmentServicer_to_server(EnvironmentRPCServer(configs), server)
    server.add_insecure_port(address)
    server.start()
    _wait_forever(server)

def _wait_forever(server):
    try:
        while True:
            time.sleep(_ONE_HOUR.total_seconds())
    except KeyboardInterrupt:
        server.stop(None)

class EnvironmentRPCServer(EnvironmentServicer):
    def __init__(self, configs):
        {setattr(self, k, v) for k, v in configs['Environment'].items()}
        self.configs = configs
        # launch!
        env_class = load_class('shiva.envs', self.configs['Environment']['type'])
        self.env = env_class(self.configs)
        self.env.reset()

    def GetSpecs(self, request: Empty, context):
        env_specs = EnvironmentSpecs()
        env_specs.observation_space = self.env.get_observation_space()
        action_space_dict = self.env.get_action_space()
        env_specs.action_space.discrete = action_space_dict['discrete']
        env_specs.action_space.param = action_space_dict['param']
        env_specs.action_space.acs_space = action_space_dict['acs_space']
        env_specs.num_instances = self.env.num_instances if hasattr(self.env, 'num_instances') else 1
        env_specs.num_agents_per_instance = self.env.num_agents_per_instance if hasattr(self.env, 'num_agents_per_instance') else 1
        return env_specs

    def Step(self, env_in: EnvStepInput, context):
        '''
            This method is for communication with the Learner at every step!
            SingleAgent but easily extendable to MultiAgent
        '''

        '''
            1. Validate correct format of input
        '''
        all_agents_instances_actions = []

        agent_ids = list(env_in.agent_actions.keys())
        if hasattr(self.env, 'num_agents'):
            if len(agent_ids) != self.env.num_agents:
                msg = 'Format error. Expecting actions for {} agents, got for {}'.format(self.env.num_agents, len(agent_ids))
                return self._return_error(EnvStepOutput, context, msg)
        for _id in agent_ids:
            agent_action = env_in.agent_actions[_id].data
            if hasattr(self.env, 'num_instances'):
                if len(agent_action) != self.env.num_instances:
                    msg = 'Format error. Expecting actions for {} instances, got for {}'.format(self.env.num_instances, len(agent_action))
                    return self._return_error(EnvStepOutput, context, msg)
            '''
                2. Convert to numpy 
                    --- This should be discussed & compatible with the environment being used!
                    dim=0       Number of instances
                    dim=1       Number of agents
            '''
            all_agents_instances_actions.append( [ agent_action[i].data for i in range(len(agent_action)) ] )

        next_obs, rews, dones, _ = self.env.step(all_agents_instances_actions[0]) # !!!!!!! SINGLE AGENT NOTATION HERE !!!!!!!

        # Prepare server response
        env_state = EnvStepOutput()

        only_agent_id = agent_ids[0] # !!!!!!! SINGLE AGENT NOTATION HERE !!!!!!!

        agent_state = env_state.agent_states[only_agent_id].data.add()
        agent_state.next_observation.data.extend(next_obs.tolist())
        agent_state.reward = self.env.reward_per_step
        agent_state.done = self.env.done

        agent_metric = env_state.agent_metrics[only_agent_id].data.add()
        agent_metric.steps_per_episode = self.env.steps_per_episode
        agent_metric.step_count = self.env.step_count
        agent_metric.temp_done_counter = self.env.temp_done_counter
        agent_metric.done_count = self.env.done_count
        agent_metric.reward_per_step = self.env.reward_per_step
        agent_metric.reward_per_episode = self.env.reward_per_episode
        agent_metric.reward_total = self.env.reward_total

        return env_state

    def Reset(self, request: Empty, context):
        self.env.reset()
        next_obs = self.env.get_observation()
        rews = self.env.get_reward()
        dones = self.env.done

        # Prepare server response
        env_output = EnvStepOutput()

        only_agent_id = '0' #agent_ids[0] # !!!!!!! SINGLE AGENT NOTATION HERE !!!!!!!

        agent_state = env_output.agent_states[only_agent_id].data.add()
        agent_state.next_observation.data.extend(next_obs.tolist())
        agent_state.reward = self.env.reward_per_step
        agent_state.done = self.env.done

        agent_metric = env_output.agent_metrics[only_agent_id].data.add()
        agent_metric.steps_per_episode = self.env.steps_per_episode
        agent_metric.step_count = self.env.step_count
        agent_metric.temp_done_counter = self.env.temp_done_counter
        agent_metric.done_count = self.env.done_count
        agent_metric.reward_per_step = self.env.reward_per_step
        agent_metric.reward_per_episode = self.env.reward_per_episode
        agent_metric.reward_total = self.env.reward_total

        return env_output

    def _return_error(self, return_class, context, msg):
        context.set_details(msg)
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        return return_class()
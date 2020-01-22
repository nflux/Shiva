import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent.parent))

from concurrent import futures
import time
import math
import logging

import grpc

from shiva.core.communication_objects.env_command_pb2 import EnvironmentCommand
from shiva.core.communication_objects.env_step_pb2 import ( EnvStepInput, EnvStepOutput )
from shiva.core.communication_objects.env_specs_pb2 import EnvironmentSpecs
from shiva.core.communication_objects.env_metrics_pb2 import AgentMetrics
from shiva.core.communication_objects.agent_state_pb2 import AgentState
from shiva.core.communication_objects.service_env_pb2_grpc import EnvironmentServicer, add_EnvironmentServicer_to_server
from shiva.core.communication_objects.helpers_pb2 import Empty

import random
import numpy as np

class Environment(EnvironmentServicer):

    def __init__(self):
        pass

    def GetSpecs(self, request: Empty, context):
        env_specs = EnvironmentSpecs()
        env_specs.observation_space = 1
        action_space_dict = 2
        env_specs.action_space.discrete = 3
        env_specs.action_space.param = 4
        env_specs.action_space.acs_space = 5
        env_specs.num_instances = 6
        env_specs.num_agents_per_instance = 7
        return env_specs

    def Step(self, EnvStepInput, context):
        '''
            Dummy example for returning current state of the env with 2 agents
        '''
        env_state = EnvStepOutput()
        n_agents = 1
        n_steps = 1

        for agent_n in range(n_agents):
            for step_n in range(n_steps):
                a = env_state.agent_states[str(agent_n)].data.add()
                a.next_observation.data.extend(np.random.rand(5))
                a.reward = np.random.rand(1)[0]
                a.done = bool(random.randint(0, 1))
            m = env_state.agent_metrics[str(agent_n)].data.add()
            m.steps_per_episode = 1
            m.step_count = 2
            m.temp_done_counter = 3
            m.done_count = 4
            m.reward_per_step = 5
            m.reward_per_episode = 6
            m.reward_total = 7
        return env_state

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_EnvironmentServicer_to_server(Environment(), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
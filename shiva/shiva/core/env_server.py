from concurrent import futures
import time
import math
import logging

import grpc

from communication_objects.all_pb2 import (
    EnvStepInput,
    EnvironmentSpecs, AgentMetrics,
    AgentState, EnvStepOutput
)
from communication_objects.all_pb2_grpc import EnvironmentServicer, add_EnvironmentServicer_to_server

import random
import numpy as np

class Environment(EnvironmentServicer):

    def __init__(self):
        pass

    def Step(self, EnvStepInput, context):
        '''
            Dummy example for returning current state of the env with 2 agents
        '''
        env_state = EnvStepOutput()

        a = env_state.agent_states['0'].data.add()
        a.next_observation.data.extend(np.random.rand(5))
        a.reward = np.random.rand(1)[0]
        a.done = bool(random.randint(0, 1))

        a = env_state.agent_states['1'].data.add()
        a.next_observation.data.extend(np.random.rand(5))
        a.reward = np.random.rand(1)[0]
        a.done = bool(random.randint(0, 1))

        m = env_state.agent_metrics['0'].data.add()
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
    add_EnvironmentServicer_to_server(
        Environment(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
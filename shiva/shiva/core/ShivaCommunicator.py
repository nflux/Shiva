import grpc
import multiprocessing
# import contextlib
# import socket
import os
import time

from shiva.envs.Environment import Environment
from shiva.envs.EnvironmentRPCClient import EnvironmentRPCClient
from shiva.envs.EnvironmentRPCServer import serve

from shiva.core.communication_objects.helpers_pb2 import Empty
from shiva.core.communication_objects.service_env_pb2_grpc import (
    EnvironmentStub, EnvironmentServicer, add_EnvironmentServicer_to_server
)

class ShivaCommunicator():
    grpc_debug = False

    def __init__(self):
        self.open_channels = []
        self.open_processes = []
        if self.grpc_debug:
            os.environ['GRPC_VERBOSITY'] = 'DEBUG'
            grpc_trace = ['connectivity_state'] # 'all', 'http', 'api', 'tcp', 'client_channel_routing', 'cares_resolver']
            os.environ['GRPC_TRACE'] = ','.join(grpc_trace)

    def start_meta_server(self, meta_id, configs):

    def start_learner_server(self, meta_id, learner_id, configs):

    def start_multienv_server(self, meta_id, multienv_id, configs):
        p = self.start_process(serve_multienv, (address, configs))
        self.open_multienv_process.append(p)

    def start_env(self, multienv_id, address, configs, need_EnvSpecs=False):
        env = SingleEnvironment()
        env.multienv_client = self.create_env2multienv_client(multienv_id, address, )
        env.learner_client = self.create_env2learner_client()

        if need_EnvSpecs:
            env.launch()
            EnvSpecs = env.get_specs()

        p = self._start_process(env.launch, args=(address, configs))
        self.open_envs_process.append(p)

        if need_EnvSpecs:
            return EnvSpecs

    def _start_process(self, target_f, args):
        p = multiprocessing.Process(target=target_f, args=args)
        p.start()
        time.sleep(1)
        # self.open_processes.append(p)
        return p

    def create_learner2meta_client(self, meta_id, address, configs):
        return self._create_cls_client(MetaRPCClient, address, configs)

    def create_multienv2learner_client(self, learner_id, address, configs):
        return self._create_cls_client(LearnerRPCClient, address, configs)

    def create_env2multienv_client(self, multienv_id, address, configs):
        return self._create_cls_client(MultiEnvironmentRPCClient, address, configs)

    def create_env2learner_client(self, learner_id, address, configs):
        return self._create_cls_client()

    def _create_cls_client(self, cls, address, configs):
        channel = self._open_new_channel(address)
        client_env = cls(channel, configs)
        return client_env

    def _open_new_channel(self, address):
        channel = grpc.insecure_channel(address)
        self.open_channels.append(channel)
        return channel

    def close_connections(self):
        for channel in self.open_channels:
            channel.close()

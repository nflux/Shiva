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

    def start_env_server(self, address, configs):
        # serve(address, configs)
        p = multiprocessing.Process(target=serve, args=(address, configs))
        p.start()
        time.sleep(2)
        self.open_processes.append(p)

    def get_learner2env_client(self, learner_id, address, configs):
        '''
            Creates and returns a EnvironmentRPCClient thru the @address

            @learner_id     who owns this connection with the environment
            @address        IP:port such as 'localhost:50051'
        '''
        channel = self._open_new_channel(address)
        client_env = EnvironmentRPCClient(channel, configs)
        return client_env

    def _open_new_channel(self, address):
        channel = grpc.insecure_channel(address)
        self.open_channels.append(channel)
        return channel

    def close_connections(self):
        for channel in self.open_channels:
            channel.close()

    # @contextlib.contextmanager
    # def reserve_port(self):
    #     '''
    #         Note: I think that this only verifies in localhost
    #     '''
    #     """Find and reserve a port for all subprocesses to use."""
    #     sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    #     if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
    #         raise RuntimeError("Failed to set SO_REUSEPORT.")
    #     sock.bind(('', 0))
    #     try:
    #         yield sock.getsockname()[1]
    #     finally:
    #         sock.close()
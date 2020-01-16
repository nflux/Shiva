import time
from concurrent import futures
import datetime
import grpc, os, json
import numpy as np
from mpi4py import MPI

from shiva.core.communication_objects.enums_pb2 import ComponentType
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage

from shiva.core.communication_objects.service_learner_pb2_grpc import LearnerServicer, add_LearnerServicer_to_server, LearnerStub

class CommMultiAgentLearnerServer(LearnerServicer):
    '''
        gRPC Server
    '''
    def __init__(self, learner_tags):
        self.learner_tags = learner_tags
        self.learner = MPI.Comm.Get_parent()
        self.status = MPI.Status()
        self.any_src, self.any_tag = MPI.ANY_SOURCE, MPI.ANY_TAG

        self.configs = None
        self.debug("MPI Request for configs")
        self.configs = self.learner.recv(None, source=0, tag=self.learner_tags.configs)
        self.debug("Received config with {} keys".format(len(self.configs.keys())))

    def SendTrajectories(self, request: SimpleMessage, context) -> Empty:
        trajectories = json.loads(request.data)
        # trajectories = np.array(json.loads(request.data))
        self.learner.send(trajectories, 0, self.learner_tags.trajectories) # this should be non-blocking MPI send
        return Empty()

    def SendSpecs(self, simple_message: SimpleMessage, context) -> Empty:
        specs = json.loads(simple_message.data)
        if specs['type'] == ComponentType.MULTIENV:
            self.learner.send(specs, 0, self.learner_tags.menv_specs)
            self.debug("gRPC received MultiEnvSpecs, and passed them to the Learner")
        return Empty()

    def debug(self, msg):
        print("PID {} LearnerServer\t\t{}".format(os.getpid(), msg))


def serve(address, shared_dict, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_LearnerServicer_to_server(CommMultiAgentLearnerServer(shared_dict), server)
    server.add_insecure_port(address)
    server.start()
    _wait_forever(server)

def _wait_forever(server):
    try:
        while True:
            # time.sleep(_ONE_HOUR.total_seconds())
            pass
    except KeyboardInterrupt:
        server.stop(None)

def get_learner_stub(address: str):
    class gRPC_LearnerStub(LearnerStub):
        '''
            gRPC Client
        '''
        def __init__(self, address):
            self.channel = grpc.insecure_channel(address)
            super(gRPC_LearnerStub, self).__init__(self.channel)

        def send_menv_specs(self, menv_specs: dict) -> None:
            simple_message = SimpleMessage()
            menv_specs['type'] = ComponentType.MULTIENV
            simple_message.data = json.dumps(menv_specs)
            response = self.SendSpecs(simple_message)
            return None

        def send_evol_config(self, evol_config: dict) -> None:
            assert "NotImplemented"
            pass

        def send_trajectory(self, trajectories: list) -> None:
            simple_message = SimpleMessage()
            simple_message.data = json.dumps(trajectories)
            response = self.SendTrajectories(simple_message)
            return None

    return gRPC_LearnerStub(address)
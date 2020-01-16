import time
from concurrent import futures
import datetime
import grpc, os, sys, json
from mpi4py import MPI

from shiva.core.communication_objects.specs_pb2 import SpecsProto, MultiEnvSpecsProto
from shiva.core.communication_objects.configs_pb2 import ConfigProto
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage
from shiva.core.communication_objects.enums_pb2 import ComponentType

from shiva.core.communication_objects.service_meta_pb2_grpc import MetaLearnerServicer, add_MetaLearnerServicer_to_server, MetaLearnerStub

class CommMultiLearnerMetaLearnerServer(MetaLearnerServicer):
    '''
        gRPC Server
    '''
    def __init__(self, meta_tags):
        self.meta_tags = meta_tags
        self.meta = MPI.Comm.Get_parent()
        self.status = MPI.Status()
        self.any_src, self.any_tag = MPI.ANY_SOURCE, MPI.ANY_TAG

        self.configs = None
        # self.debug("MPI Request for configs")
        self.configs = self.meta.recv(None, source=0, tag=self.meta_tags.configs)
        self.debug("Received config with {} keys".format(len(self.configs.keys())))

        self.menvs_check = []
        self.learners_data = []

    def GetConfigs(self, request: SimpleMessage, context) -> SimpleMessage:
        response = SimpleMessage()
        response.data = json.dumps(self.configs)
        return response

    def SendSpecs(self, request: SimpleMessage, context):
        specs = json.loads(request.data)
        if specs['type'] == ComponentType.MULTIENV:
            self.menvs_check.append(specs['id'])
            # self.debug("received gRPC MultiEnvSpecs and sending MPI to Meta")
            self.meta.send(specs, 0, self.meta_tags.menv_specs)
        elif specs['type'] == ComponentType.LEARNER:
            self.learners_data.append(specs)
            # self.debug("received gRPC LearnerSpecs and sending MPI to Meta")
            self.meta.send(specs, 0, self.meta_tags.learner_specs)
        else:
            self._return_error(SpecsProto, context, "InvalidComponentType")
        return Empty()

    def debug(self, msg):
        print("PID {} MetaServer\t\t{}".format(os.getpid(), msg))

def serve(address, meta_tags, max_workers=5):
    '''
        Start gRPC server
    '''
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_MetaLearnerServicer_to_server(CommMultiLearnerMetaLearnerServer(meta_tags), server)
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

def get_meta_stub(address: str):
    class gRPC_MetaLearnerStub(MetaLearnerStub):
        '''
            gRPC Client
        '''
        def __init__(self, address):
            self.channel = grpc.insecure_channel(address)
            super(gRPC_MetaLearnerStub, self).__init__(self.channel)

        def get_configs(self) -> dict:
            msg = SimpleMessage()
            simple_message = self.GetConfigs(msg, wait_for_ready=True)
            return json.loads(simple_message.data)

        def send_menv_specs(self, menv_specs: dict) -> None:
            menv_specs['type'] = ComponentType.MULTIENV
            simple_message = SimpleMessage()
            simple_message.data = json.dumps(menv_specs)
            response = self.SendSpecs(simple_message)
            return None

        def send_learner_specs(self, learner_specs: dict) -> None:
            learner_specs['type'] = ComponentType.LEARNER
            simple_message = SimpleMessage()
            simple_message.data = json.dumps(learner_specs)
            response = self.SendSpecs(simple_message)
            return None

    return gRPC_MetaLearnerStub(address)
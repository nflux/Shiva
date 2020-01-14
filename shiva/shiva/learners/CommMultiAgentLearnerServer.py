import time
import futures
import datetime
import grpc

from shiva.core.communication_objects.configs_pb2 import EvolutionConfigProto
from shiva.core.communication_objects.env_step_pb2 import TrajectoriesProto
from shiva.core.communication_objects.env_specs_pb2 import MultiEnvSpecsProto
from shiva.core.communication_objects.helpers_pb2 import Empty
from shiva.core.communication_objects.configs_pb2 import EvolutionConfigProto

from shiva.core.communication_objects.service_learner_pb2 import LearnerServicer, add_LearnerServicer_to_server, LearnerStub

from shiva.helpers.grpc_utils import (
    from_dict_2_MultiEnvSpecsProto, from_MultiEnvSpecsProto_2_dict,
    from_dict_2_EvolutionConfigProto, from_EvolutionConfigProto_2_dict,
    from_dict_2_TrajectoriesProto, from_TrajectoriesProto_2_dict
)

class CommMultiAgentLearnerServer(LearnerServicer):
    '''
        gRPC Server
    '''
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict

    def SendMultiEnvSpecs(self, menv_specs_proto: MultiEnvSpecsProto, context) -> Empty:
        self.shared_dict['menv_specs'] = from_MultiEnvSpecsProto_2_dict(menv_specs_proto)
        return Empty()

    def SendEvolutionConfig(self, evol_config_proto: EvolutionConfigProto, context) -> Empty:
        self.shared_dict['evol_config'] = from_EvolutionConfigProto_2_dict(evol_config_proto)
        return Empty()

    def SendTrajectories(self, trajectories_proto: TrajectoriesProto, context) -> Empty:
        self.shared_dict['trajectories_queue'].push(from_TrajectoriesProto_2_dict(trajectories_proto))
        return Empty()

def start_learner_server(address, shared_dict, max_workers=5):
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
            super(gRPC_LearnerStub, self).__init__(address)

        def send_menv_specs(self, menv_specs: dict) -> None:
            empty_msg = self.SendMultiEnvSpecs(from_dict_2_MultiEnvSpecsProto(menv_specs))
            return None

        def send_evol_config(self, evol_config: dict) -> None:
            empty_msg = self.SendEvolutionConfig(from_dict_2_EvolutionConfigProto(evol_config))
            return None

        def send_trajectory(self, trajectories: dict) -> None:
            empty_msg = self.SendTrajectories(from_dict_2_TrajectoriesProto(trajectories))
            return None

    return LearnerStub(address)
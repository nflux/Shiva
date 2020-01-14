import time
import futures
import datetime
import grpc

from shiva.core.communication_objects.env_specs_pb2 import MultiEnvSpecsProto
from shiva.core.communication_objects.helpers_pb2 import Empty, StringMessage
from shiva.core.communication_objects.metrics_pb2 import TrainingMetricsProto, EvaluationMetricsProto
from shiva.core.communication_objects.configs_pb2 import StatusProto, ComponentType

from shiva.core.communicator_objects.service_meta_pb2_grpc import MetaLearnerServicer, add_MetaLearnerServicer_to_server, MetaLearnerStub

from shiva.helpers.grpc_utils import (
    from_dict_2_StatusProto, from_StatusProto_2_dict
    from_dict_2_MultiEnvSpecsProto, from_MultiEnvSpecsProto_2_dict,
    from_dict_2_TrainingMetricsProto, from_TrainingMetricsProto_2_dict,
    from_dict_2_EvolutionMetricProto, from_EvolutionMetricProto_2_dict
)

class CommMultiLearnerMetaLearnerServer(MetaLearnerServicer):
    '''
        gRPC Server
    '''
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict

    def SendStatus(self, status_proto: StatusProto, context) -> Empty:
        status = from_StatusProto_2_dict(status_proto)
        if status['type'] == ComponentType.LEARNER:
            self.shared_dict['learners_info'][status.id] = status
        elif status['type'] == ComponentType.MULTIENV:
            self.shared_dict['menvs_info'][status.id] = status
        elif status['type'] == ComponentType.EVAL:
            pass

    def SendMultiEnvSpecs(self, menv_specs_proto: MultiEnvSpecsProto, context) -> Empty:
        self.shared_dict['menv_specs'] = from_MultiEnvSpecsProto_2_dict(menv_specs_proto)
        return Empty()

    def SendTrainingMetrics(self, train_metrics_proto: TrainingMetricsProto, context) -> Empty:
        self.shared_dict['train_metrics'] = from_TrainingMetricsProto_2_dict(train_metrics_proto)
        return Empty()

    def SendEvaluationMetrics(self, eval_metrics_proto: EvaluationMetricsProto, context) -> Empty:
        self.shared_dict['eval_config'] = from_EvolutionMetricProto_2_dict(eval_metrics_proto)
        return Empty()

def start_meta_server(address, shared_dict, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_MetaLearnerServicer_to_server(CommMultiLearnerMetaLearnerServer(shared_dict), server)
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
            super(gRPC_MetaLearnerStub, self).__init__(address)

        def send_menv_specs(self, menv_specs: dict) -> None:
            response = self.SendMultiEnvSpecs(from_dict_2_MultiEnvSpecsProto(menv_specs))
            return None

        def send_train_metrics(self, train_metrics: dict):
            response = self.SendTrainingMetrics(from_dict_2_TrainingMetricsProto(train_metrics))
            return None

        def send_eval_metrics(self, eval_metrics: dict) -> None:
            response = self.SendEvaluationMetrics(from_dict_2_EvolutionMetricProto(eval_metrics))
            return None

    return gRPC_MetaLearnerStub(address)
import grpc
from concurrent import futures

from shiva.core.communication_objects.configs_pb2 import NewAgentsConfigProto
from shiva.core.communication_objects.env_specs_pb2 import EnvSpecsProto, LearnersInfoProto
from shiva.core.communication_objects.env_step_pb2 import ActionsProto, ObservationsProto
from shiva.core.communication_objects.service_multienv_pb2_grpc import MultiEnvironmentServicer, add_MultiEnvironmentServicer_to_server, MultiEnvironmentStub
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage

from shiva.helpers.grpc_utils import (
    from_dict_2_ObservationsProto, from_ObservationsProto_2_dict,
    from_dict_2_ActionsProto, from_ActionsProto_2_dict,
    from_dict_2_NewAgentsConfigProto, from_NewAgentsConfigProto_2_dict,
    from_dict_2_EnvSpecsProto, from_EnvSpecsProto_2_dict,
    from_dict_2_SimpleMessage, from_SimpleMessage_2_int, from_SimpleMessage_2_string,
    from_dict_2_LearnersInfoProto, from_LearnersInfoProto_2_dict
)

class MultiEnvironmentServer(MultiEnvironmentServicer):
    '''
        gRPC Server
    '''

    def __init__(self, shared_dict):
        self.shared_dict = shared_dict

    def SendObservations(self, observation_proto: ObservationsProto, context) -> ActionsProto:
        observations, step_count = from_ObservationsProto_2_dict()
        actions = {}
        for agent_id, obs in observations.items():
            actions[agent_id] = self.shared_dict['agents'].get_action(obs, step_count)
        return from_dict_2_ActionsProto(actions)

    def SendNewAgents(self, new_agent_config_proto: NewAgentsConfigProto, context) -> Empty:
        assert "NotImplemented"
        agents_config = from_NewAgentsConfigProto_2_dict(new_agent_config_proto)
        agents = {}
        # load agents networks
        # ....
        # ....
        self.shared_dict['agents'] = agents
        return Empty()

    def SendEnvSpecs(self, env_specs_proto: EnvSpecsProto, context) -> Empty:
        self.shared_dict['env_specs'] = from_EnvSpecsProto_2_dict(env_specs_proto)
        return Empty()

    def GetLearnersInfo(self, simple_msg: SimpleMessage, context) -> LearnersInfoProto:
        env_id = from_SimpleMessage_2_int(simple_msg)
        while self.shared_dict['learners_info'] == None: # we dont have the data yet
            pass
        return from_dict_2_LearnersInfoProto(self.shared_dict['learners_info'][env_id])

def serve_multienv(address, shared_dict, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_MultiEnvironmentServicer_to_server(MultiEnvironmentServer(shared_dict), server)
    server.add_insecure_port(address)
    server.start()
    _wait_forever(server)

def _wait_forever(server):
    try:
        while True:
            # time.sleep(_ONE_DAY.total_seconds())
            pass
    except KeyboardInterrupt:
        server.stop(None)

def get_menv_stub(address: str):
    class gRPC_MultiEnvironmentStub(MultiEnvironmentStub):
        '''
            gRPC Client
        '''
        def __init__(self, address):
            super(gRPC_MultiEnvironmentStub, self).__init__(address)

        def send_observations(self, observations: dict) -> dict:
            actions =  self.SendObservations(from_dict_2_ObservationsProto(observations))
            return from_ActionsProto_2_dict(actions)

        def send_new_agents(self, new_agents: dict) -> dict:
            new_agents_proto = self.SendNewAgents(from_dict_2_NewAgentsConfigProto(new_agents))
            return from_NewAgentsConfigProto_2_dict(new_agents_proto)

        def send_env_specs(self, env_specs: dict) -> dict:
            env_specs_proto = self.SendEnvSpecs(from_dict_2_EnvSpecsProto(env_specs))
            return from_EnvSpecsProto_2_dict(env_specs_proto)

        def get_learners_address(self, env_id: int) -> dict:
            learners_info_proto = self.GetLearnersInfo(from_dict_2_SimpleMessage(env_id))
            return from_LearnersInfoProto_2_dict(learners_info_proto)

    return MultiEnvironmentStub(address)
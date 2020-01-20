import grpc, os, json
import torch
from concurrent import futures
from mpi4py import MPI

from shiva.core.admin import Admin

from shiva.core.communication_objects.enums_pb2 import ComponentType, LoadType
from shiva.core.communication_objects.configs_pb2 import ConfigProto
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage
from shiva.core.communication_objects.service_multienv_pb2_grpc import MultiEnvironmentServicer, add_MultiEnvironmentServicer_to_server, MultiEnvironmentStub

class MultiEnvironmentServer(MultiEnvironmentServicer):
    '''
        gRPC Server
    '''

    def __init__(self, menv_tags):
        self.menv_tags = menv_tags
        self.menv = MPI.Comm.Get_parent()
        self.status = MPI.Status()
        self.any_src, self.any_tag = MPI.ANY_SOURCE, MPI.ANY_TAG

        # self.debug("MPI Request for Configs")
        self.configs = self.menv.recv(None, source=0, tag=self.menv_tags.configs)
        self.debug('Received config with {} keys'.format(len(self.configs.keys())))

        self.envs_data = []
        self.learners_data = []

        self.step_count = 0

    def GetActions(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        observations = json.loads(simple_message.data)
        '''
            Single Agent assumption here!
            And having Agent locally..
        '''
        # check if we have a new agent to load..
        actions = self.agents[0].get_action(observations, self.step_count)
        # self.debug("{}".format(actions))
        response = SimpleMessage()
        response.data = json.dumps(actions)
        self.step_count += 1
        # self.debug("Received Agent Step {}".format(self.agents[0].step_count))
        return response

    def SendSpecs(self, simple_message: SimpleMessage, context) -> Empty:
        specs = json.loads(simple_message.data)
        if specs['type'] == ComponentType.ENVIRONMENT:
            self.envs_data.append(specs) # this might be unnecessary to accumulate?
            self.debug("received gRPC EnvSpecs {} and sending MPI to MultiEnv".format(specs['id']))
            self.menv.send(specs, 0, self.menv_tags.env_specs)
        elif specs['type'] == ComponentType.LEARNER:
            self.learners_data.append(specs)
            self.debug("received gRPC LearnerSpecs {} and sharing with MultiEnv".format(specs['id']))
            '''
                Loading Agents locally for now!
            '''
            self.agents = Admin._load_agents(specs['load_path'])
            self.menv.send(specs, 0, self.menv_tags.learner_specs)
        else:
            self._return_error(SimpleMessage, context, "InvalidComponentType")
        return Empty()

    def GetSpecs(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        response = SimpleMessage()
        specs = json.loads(simple_message.data)
        if specs['type'] == ComponentType.ENVIRONMENT:
            response.data = json.dumps(self.learners_data)
        else:
            self._return_error(SimpleMessage, context, "InvalidComponentType")
        return response

    def SendConfig(self, config_proto: ConfigProto, context) -> Empty:
        config = {
            'load_type': config_proto.load_type,
            'load_path': config_proto.load_path
        }
        if config_proto.type == ComponentType.AGENTS:
            '''
                Loading Single Agent locally for now!
            '''
            self.agents = Admin._load_agents(config['load_path'])
            # self.debug("Received Agent Step {}".format(self.agents[0].step_count))
            # self.menv.send(config, 0, self.menv_tags.new_agents) # if want to share with the MultiEnv
        else:
            self._return_error(ConfigProto, context, "InvalidComponentType")
        return Empty()

    def _return_error(self, return_class, context, msg):
        context.set_details(msg)
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        return return_class()

    def debug(self, msg):
        print("PID {} MultiEnvServer\t\t{}".format(os.getpid(), msg))

    # def SendObservations(self, request, context) -> ActionsProto:
    #     assert "NotImplemented"

def serve(address, menv_tags, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers,), options=options)
    add_MultiEnvironmentServicer_to_server(MultiEnvironmentServer(menv_tags), server)
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
            self.channel = grpc.insecure_channel(address)
            super(gRPC_MultiEnvironmentStub, self).__init__(self.channel)

        def get_actions(self, observations) -> dict:
            simple_message = SimpleMessage()
            simple_message.data = json.dumps(list(observations))
            response = self.GetActions(simple_message, wait_for_ready=True)
            # print(response.data, type(response.data))
            return json.loads(response.data)

        def send_env_specs(self, env_specs: dict) -> None:
            simple_message = SimpleMessage()
            env_specs['type'] = ComponentType.ENVIRONMENT
            simple_message.data = json.dumps(env_specs)
            response = self.SendSpecs(simple_message)
            return None

        def send_learner_specs(self, specs: dict) -> None:
            '''
                Methods used by the Learners
            '''
            simple_message = SimpleMessage()
            specs['type'] = ComponentType.LEARNER
            simple_message.data = json.dumps(specs)
            response = self.SendSpecs(simple_message)
            return None

        def get_learners_specs(self, spec: dict) -> dict:
            '''
                Method used by the single Environment
            '''
            simple_message = SimpleMessage()
            spec['type'] = ComponentType.ENVIRONMENT
            simple_message.data = json.dumps(spec)
            response_simple_message = self.GetSpecs(simple_message)
            return json.loads(response_simple_message.data)

        def send_new_agents(self, checkpoint_url: str) -> None:
            '''
                Method used by the Learners
            '''
            config_proto = ConfigProto()
            config_proto.type = ComponentType.AGENTS
            config_proto.load_type = LoadType.LOCAL
            config_proto.load_path = checkpoint_url
            response = self.SendConfig(config_proto)
            return None

    return gRPC_MultiEnvironmentStub(address)
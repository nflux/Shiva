import grpc, os, json
import torch
from concurrent import futures
from mpi4py import MPI

from shiva.core.admin import Admin

from shiva.core.communication_objects.enums_pb2 import ComponentType, LoadType
from shiva.core.communication_objects.configs_pb2 import ConfigProto
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage
from shiva.core.communication_objects.service_multienv_pb2_grpc import IOHandlerServicer, add_IOHandlerServicer_to_server, IOHandlerStub

class IOHandlerServer(IOHandlerServicer):
    '''
        gRPC Server
    '''

    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    def __init__(self):
        self.configs = self.menv.bcast(None, source=0, tag=self.menv_tags.configs)
        self.debug('Received config with {} keys'.format(len(self.configs.keys())))

        self._urls_used = set()

    def AskIORequest(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        # response if directory is available
        msg = json.loads(simple_message.data)
        req_spec, req_url = msg['spec'], msg['url']
        _has_access = req_url not in self._urls_used
        if _has_access:
            self._urls_used.add(req_url)
        response = SimpleMessage()
        response.data = json.dumps({'spec': msg['spec'], 'has_access': _has_access})
        return response

    def DoneIO(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        msg = json.loads(simple_message.data)
        req_spec, req_url = msg['spec'], msg['url']
        self._urls_used.remove(req_spec)

def get_io_stub(address: str):
    class gRPC_IOHandlerStub(IOHandlerStub):
        '''
            gRPC Client
        '''

        def __init__(self, address):
            self.channel = grpc.insecure_channel(address)
            super(gRPC_IOHandlerStub, self).__init__(self.channel)

        def request_io(self, spec, url):
            simple_message = SimpleMessage()
            data = {'spec': spec, 'url': url}
            simple_message.data = json.dumps(data)
            response = self.AskIORequest(simple_message, wait_for_ready=True) # check if is possible to include a timeout
            has_access = False
            if response is not None:
                msg_back = json.loads(response.data)
                spec_back, has_access = msg_back['spec'], msg_back['has_access']
            return has_access

        def done_io(self, spec, url):
            simple_message = SimpleMessage()
            data = {'spec': spec, 'url': url}
            simple_message.data = json.dumps(data)
            response = self.DoneIO(simple_message) # check if is possible to include a timeout
            return None

    return gRPC_IOHandlerStub(address)


'''Starting functions of the gRPC server'''

def serve(address, configs, max_workers=5):
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers, ), options=options)
    add_IOHandlerServicer_to_server(IOHandlerServer(configs), server)
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



        # self.envs_data = []
        # self.learners_data = []

        # self.step_count = 0

    # def GetActions(self, simple_message: SimpleMessage, context) -> SimpleMessage:
    #     observations = json.loads(simple_message.data)
    #     '''
    #         Single Agent assumption here!
    #         And having Agent locally..
    #     '''
    #     # check if we have a new agent to load..
    #     actions = self.agents[0].get_action(observations, self.step_count)
    #     # self.debug("{}".format(actions))
    #     response = SimpleMessage()
    #     response.data = json.dumps(actions)
    #     self.step_count += 1
    #     # self.debug("Received Agent Step {}".format(self.agents[0].step_count))
    #     return response
    #
    # def SendSpecs(self, simple_message: SimpleMessage, context) -> Empty:
    #     specs = json.loads(simple_message.data)
    #     if specs['type'] == ComponentType.ENVIRONMENT:
    #         self.envs_data.append(specs) # this might be unnecessary to accumulate?
    #         self.debug("received gRPC EnvSpecs {} and sending MPI to MultiEnv".format(specs['id']))
    #         self.menv.send(specs, 0, self.menv_tags.env_specs)
    #     elif specs['type'] == ComponentType.LEARNER:
    #         self.learners_data.append(specs)
    #         self.debug("received gRPC LearnerSpecs {} and sharing with MultiEnv".format(specs['id']))
    #         '''
    #             Loading Agents locally for now!
    #         '''
    #         self.agents = Admin._load_agents(specs['load_path'])
    #         self.menv.send(specs, 0, self.menv_tags.learner_specs)
    #     else:
    #         self._return_error(SimpleMessage, context, "InvalidComponentType")
    #     return Empty()
    #
    # def GetSpecs(self, simple_message: SimpleMessage, context) -> SimpleMessage:
    #     response = SimpleMessage()
    #     specs = json.loads(simple_message.data)
    #     if specs['type'] == ComponentType.ENVIRONMENT:
    #         response.data = json.dumps(self.learners_data)
    #     else:
    #         self._return_error(SimpleMessage, context, "InvalidComponentType")
    #     return response
    #
    # def SendConfig(self, configs_proto: ConfigProto, context) -> Empty:
    #     configs = {
    #         'load_type': configs_proto.load_type,
    #         'load_path': configs_proto.load_path
    #     }
    #     if configs_proto.type == ComponentType.AGENTS:
    #         '''
    #             Loading Single Agent locally for now!
    #         '''
    #         self.new_agents_config = configs
    #         self.agents = Admin._load_agents(self.new_agents_config['load_path'])
    #         # self.debug("Received Agent Step {}".format(self.agents[0].step_count))
    #         # self.menv.send(config, 0, self.menv_tags.new_agents) # if want to share with the MultiEnv
    #     else:
    #         self._return_error(ConfigProto, context, "InvalidComponentType")
    #     return Empty()
    #
    # def _return_error(self, return_class, context, msg):
    #     context.set_details(msg)
    #     context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
    #     return return_class()
    #
    # def debug(self, msg):
    #     print("PID {} MultiEnvServer\t\t{}".format(os.getpid(), msg))
    #
    # # def SendObservations(self, request, context) -> ActionsProto:
    # #     assert "NotImplemented"



        ''''''

        # def get_actions(self, observations) -> dict:
        #     simple_message = SimpleMessage()
        #     simple_message.data = json.dumps(list(observations))
        #     response = self.GetActions(simple_message, wait_for_ready=True)
        #     # print(response.data, type(response.data))
        #     return json.loads(response.data)
        #
        # def send_env_specs(self, env_specs: dict) -> None:
        #     simple_message = SimpleMessage()
        #     env_specs['type'] = ComponentType.ENVIRONMENT
        #     simple_message.data = json.dumps(env_specs)
        #     response = self.SendSpecs(simple_message)
        #     return None
        #
        # def send_learner_specs(self, specs: dict) -> None:
        #     '''
        #         Methods used by the Learners
        #     '''
        #     simple_message = SimpleMessage()
        #     specs['type'] = ComponentType.LEARNER
        #     simple_message.data = json.dumps(specs)
        #     response = self.SendSpecs(simple_message)
        #     return None
        #
        # def get_learners_specs(self, spec: dict) -> dict:
        #     '''
        #         Method used by the single Environment
        #     '''
        #     simple_message = SimpleMessage()
        #     spec['type'] = ComponentType.ENVIRONMENT
        #     simple_message.data = json.dumps(spec)
        #     response_simple_message = self.GetSpecs(simple_message)
        #     return json.loads(response_simple_message.data)
        #
        # def send_new_agents(self, checkpoint_url: str) -> None:
        #     '''
        #         Method used by the Learners
        #     '''
        #     config_proto = ConfigProto()
        #     config_proto.type = ComponentType.AGENTS
        #     config_proto.load_type = LoadType.LOCAL
        #     config_proto.load_path = checkpoint_url
        #     response = self.SendConfig(config_proto)
        #     return None


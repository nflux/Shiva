import grpc, json, logging
from concurrent import futures
from mpi4py import MPI

from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage
from shiva.core.communication_objects.service_iohandler_pb2_grpc import IOHandlerServicer, add_IOHandlerServicer_to_server, IOHandlerStub

class IOHandlerServer(IOHandlerServicer):
    '''
        gRPC Server
    '''

    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    def __init__(self, configs):
        self.configs = configs
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

if __name__ == '__main__':
    logging.basicConfig()
    serve()
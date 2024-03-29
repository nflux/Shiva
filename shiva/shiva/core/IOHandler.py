import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

import threading
import grpc, json, argparse
from concurrent import futures
from mpi4py import MPI
import numpy as np

from shiva.core.admin import logger
from shiva.core.communication_objects.helpers_pb2 import Empty, SimpleMessage
from shiva.core.communication_objects.service_iohandler_pb2_grpc import IOHandlerServicer, add_IOHandlerServicer_to_server, IOHandlerStub
from shiva.helpers.utils.Tags import Tags


class IOHandlerServer(IOHandlerServicer):
    """
    IOHandler Server class in order to avoid I/O access errors when saving and loading models.
    Every Shiva component that need IO access will need to instantiate a stub in order to communicate with the IOHandler.
    This communication is based on gRPC protocol.
    The IOHandler will prevent simultaneous I/O access from the components using specific directories.
    """
    # for future MPI child abstraction
    meta = MPI.COMM_SELF.Get_parent()
    # id = MPI.COMM_SELF.Get_parent().Get_rank()
    info = MPI.Status()

    def __init__(self, stop_event):
        self._stop_event = stop_event
        self.configs = self.meta.bcast(None, root=0)
        self.log('Received configs with {} keys'.format(len(self.configs.keys())), verbose_level=1)
        self._urls_used = set()
        self.is_running = True
        t = threading.Thread(None, self._listen_2_meta)
        t.start()

    def AskIORequest(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        # response if directory is available
        msg = json.loads(simple_message.data)
        req_spec, req_url = msg['spec'], msg['url']
        _has_access = req_url not in self._urls_used
        if _has_access:
            self._urls_used.add(req_url)
        response = SimpleMessage()
        response.data = json.dumps({'spec': msg['spec'], 'has_access': _has_access})

        self.log("{} Access for {}:{} @ {}".format(_has_access, req_spec['type'], req_spec['id'], req_url), verbose_level=2)
        return response

    def DoneIO(self, simple_message: SimpleMessage, context) -> SimpleMessage:
        msg = json.loads(simple_message.data)
        req_spec, req_url = msg['spec'], msg['url']
        self.log("DoneIO of {}:{} for {}".format(req_spec['type'], req_spec['id'], req_url), verbose_level=3)
        self._urls_used.remove(req_url)
        return Empty()

    def _listen_2_meta(self):
        while self.is_running:
            if self.meta.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.close, status=self.info):
                # one of the Learners run the close()
                _ = self.meta.recv(None, source=self.info.Get_source(), tag=Tags.close)
                self.log("Close signal", verbose_level=1)
                self.is_running = False
                self.meta.Disconnect()
                self._stop_event.set()

    def log(self, msg, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['IOHandler']:
            text = '{}\t\t{}'.format(str(self), msg)
            logger.info(text, to_print=self.configs['Admin']['print_debug'])

    def __str__(self):
        return "<IOH>"

def get_io_stub(*args, **kwargs):

    class gRPC_IOHandlerStub(IOHandlerStub):
        def __init__(self, configs):
            """
            gRPC client to request IO Access

            Args:
                configs (Dict): a copy of the global config being used in the Shiva run
            """
            self.configs = configs
            self.channel = grpc.insecure_channel(self.configs['Admin']['iohandler_address'])
            super(gRPC_IOHandlerStub, self).__init__(self.channel)

        def request_io(self, spec, url, wait_for_access=False):
            has_access = False
            simple_message = SimpleMessage()
            data = {'spec': spec, 'url': url}
            # self.log("{}:{} for {}".format(spec['type'], spec['id'], url), verbose_level=3)
            simple_message.data = json.dumps(data, default=myconverter)
            _request_condition = lambda x=None: not has_access if wait_for_access else False
            while _request_condition():
                response = self.AskIORequest(simple_message) # check if is possible to include a timeout
                # self.log("Got {} response".format(response), verbose_level=2)
                if response is not None:
                    msg_back = json.loads(response.data)
                    spec_back, has_access = msg_back['spec'], msg_back['has_access']
            return has_access

        def done_io(self, spec, url):
            simple_message = SimpleMessage()
            data = {'spec': spec, 'url': url}
            simple_message.data = json.dumps(data, default=myconverter)
            response = self.DoneIO(simple_message) # check if is possible to include a timeout
            return None

        def log(self, msg, verbose_level=-1):
            '''If verbose_level is not given, by default will log'''
            if verbose_level <= self.configs['Admin']['log_verbosity']['IOHandler']:
                text = '{}\t\t{}'.format(str(self), msg)
                logger.info(text, to_print=self.configs['Admin']['print_debug'])

        def __str__(self):
            return "<IOStub>"

    return gRPC_IOHandlerStub(*args, **kwargs)


'''Starting functions of the gRPC server'''

def serve_iohandler(address, server_args=(), max_workers=5):
    """
    Starting function of the gRPC server

    Args:
        address (str): address where the IO Handler will be accessible
        server_args:
        max_workers:

    Returns:
        None
    """
    stop_event = threading.Event()
    options = (('grpc.so_reuseport', 1),)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers, ), options=options)
    add_IOHandlerServicer_to_server(IOHandlerServer(stop_event), server)
    server.add_insecure_port(address)
    server.start()
    stop_event.wait()
    server.stop(grace=None)
    exit(0)

# def _wait_forever(server):
#     try:
#         while True:
#             # time.sleep(_ONE_DAY.total_seconds())
#             pass
#     except KeyboardInterrupt:
#         server.stop(None)

def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # elif isinstance(obj, datetime.datetime):
    #     return obj.__str__()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", required=True, type=str, help='Address where the server will be hosted')
    args = parser.parse_args()
    serve_iohandler(args.address)
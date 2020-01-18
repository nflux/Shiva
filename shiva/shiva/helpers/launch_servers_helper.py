import sys, os, argparse, time, socket
from collections import namedtuple
from pathlib import Path

import torch.multiprocessing as mp
from mpi4py import MPI
comm = MPI.COMM_WORLD

'''
    Tags are used for MPI communication between the component and it's server 
'''

# MetaLearner & MetaServer
meta_server_tags = {'close': 0, 'error': 1, 'configs': 2, 'learner_specs': 3, 'menv_specs': 4, 'train_metrics': 5}
TagsTuple = namedtuple('Tags', list(meta_server_tags.keys()))
meta_tags = TagsTuple(*list(meta_server_tags.values()))

# Learner & LearnerServer
learner_server_tags = {'close': 0, 'error': 1, 'configs': 2, 'learner_specs': 3, 'menv_specs': 4, 'trajectories': 5, 'trajectories_length': 6}
TagsTuple = namedtuple('Tags', list(learner_server_tags.keys()))
learner_tags = TagsTuple(*list(learner_server_tags.values()))

# MultiEnv & MultiEnvServer
menv_server_tags = {'close': 0, 'error': 1, 'configs': 2, 'learner_specs': 3, 'env_specs': 4, 'new_agents': 5}
TagsTuple = namedtuple('Tags', list(menv_server_tags.keys()))
menv_tags = TagsTuple(*list(menv_server_tags.values()))


def check_port(port):
    """
    Attempts to bind to the requested communicator port, checking if it is already in use.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("localhost", int(port)))
    except socket.error:
        assert 'Port not available'
    finally:
        s.close()

def start_meta_server(meta_address, maxprocs=1):
    global meta_tags
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    return _start_server(ComponentType.META, meta_address, maxprocs), meta_tags

def start_learner_server(learner_address, maxprocs=1):
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    return _start_server(ComponentType.LEARNER, learner_address, maxprocs), learner_tags

def start_menv_server(menv_address, maxprocs=1):
    global menv_tags
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    return _start_server(ComponentType.MULTIENV, menv_address, maxprocs), menv_tags

def _start_server(type, address, maxprocs=1):
    '''
        type        meta | learner | menv
    '''
    args = [__file__, '-t', str(type), '-a', address]
    # p = mp.Process(target=launch, args=(type, address))
    # p.start()
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=maxprocs)
    return comm

def launch(type, address):
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    from shiva.metalearners.CommMultiLearnerMetaLearnerServer import serve as serve_meta
    from shiva.learners.CommMultiAgentLearnerServer import serve as serve_learner
    from shiva.envs.CommMultiEnvironmentServer import serve as serve_menv

    try:
        # start the gRPC servers
        if args.type == ComponentType.META:
            serve_meta(args.address, meta_tags)
        elif args.type == ComponentType.LEARNER:
            pass
            serve_learner(args.address, learner_tags)
        elif args.type == ComponentType.MULTIENV:
            pass
            serve_menv(args.address, menv_tags)
        else:
            assert "NotValidComponentType"
    finally:
        parent = MPI.Comm.Get_parent()
        parent.Disconnect()
        sys.exit(0)

if __name__ == "__main__":
    '''
        When MPI spawns, it starts here
    '''
    sys.path.append(str(Path(__file__).absolute().parent.parent.parent)) # necessary to add Shiva modules to the path
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, type=int, help='Type of component to start')
    parser.add_argument("-a", "--address", required=True, type=str, help='Address where the server will be hosted')
    args = parser.parse_args()
    launch(args.type, args.address)
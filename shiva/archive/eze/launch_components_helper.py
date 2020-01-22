import sys, os, argparse, time
from collections import namedtuple
from pathlib import Path

from mpi4py import MPI
comm = MPI.COMM_WORLD

def start_learner(learner_id, meta_address, maxprocs=1):
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    return _start_component(ComponentType.LEARNER, learner_id, meta_address, maxprocs)

def start_menv(menv_id, meta_address, maxprocs=1):
    from shiva.core.communication_objects.enums_pb2 import ComponentType
    return _start_component(ComponentType.MULTIENV, menv_id, meta_address, maxprocs)

def start_env(menv_address):
    pass

def _start_component(type, id, parent_address, maxprocs):
    '''
        type        meta | learner | menv
    '''
    args = [__file__, '-t', str(type), '-id', str(id), '-pa', parent_address]
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=maxprocs)
    return comm

if __name__ == "__main__":
    '''
        When MPI __file__ spawns, it starts here
    '''
    sys.path.append(str(Path(__file__).absolute().parent.parent.parent)) # necessary to add Shiva modules to the path
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", required=True, type=int, help='Type of component to start')
    parser.add_argument("-id", "--new-id", required=True, type=int, help='Component ID')
    parser.add_argument("-pa", "--parent-address", required=True, type=str, help='Parent address')
    args = parser.parse_args()

    from shiva.core.communication_objects.enums_pb2 import ComponentType

    try:
        # start the individual component
        if args.type == ComponentType.LEARNER:
            from shiva.learners.CommMultiAgentLearner import CommMultiAgentLearner
            learner = CommMultiAgentLearner(args.new_id)
            learner.launch(args.parent_address)
        elif args.type == ComponentType.MULTIENV:
            from shiva.envs.CommMultiEnvironment import CommMultiEnvironment
            menv = CommMultiEnvironment(args.new_id)
            menv.launch(args.parent_address)
        else:
            assert "NotValidComponentType"
    finally:
        pass
        # print('finally')
        # parent = MPI.Comm.Get_parent()
        # parent.Disconnect()
        # sys.exit(0)

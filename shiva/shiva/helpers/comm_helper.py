
from mpi4py import MPI
comm = MPI.COMM_WORLD

def start_meta_server(script):
    comm_meta_server = MPI.COMM_SELF.Spawn(sys.executable, args=[script], maxprocs=1)
    return comm_meta_server

if __name__ == "__main__":
    from pathlib import Path
    sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
    serve("localhost:50000")
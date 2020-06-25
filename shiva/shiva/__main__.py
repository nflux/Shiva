import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

# import torch.multiprocessing as mp
# try:
#     mp.set_start_method('spawn')
# except RuntimeError:
#     pass

import main
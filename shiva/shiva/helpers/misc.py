import sys
import time
import traceback, warnings
import numpy as np
import torch
import random
import subprocess
import platform



def set_seed(seed: int):
    """
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079/7
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def handle_package(package, class_name):
    '''
        This function is used by the parse_functions()
        
        Input
            @func_str       string name of a function     g.e. "ReLU"
        Return
            Function definition object (not instantiated)       g.e. nn.ReLU
    '''
    return getattr(package, class_name, None)


def action2one_hot(action_space: int, action_idx: int, numpy: bool=True) -> np.ndarray:
    """
    Returns a one hot encoded numpy array

    Args:
        action_space (int): how many dimensions the action space has
        action_idx (int): the index we are inserting a 1 on the resulting array
        numpy (bool): If True, return a numpy array, else returns a python list

    Returns:
        None
    """
    z = np.zeros(action_space)
    z[action_idx] = 1
    return z if numpy else list(z)


def action2one_hot_v(action_space: int, action_idx: int) -> torch.tensor:
    """
    Returns a one hot encoded torch tensor

    Args:
        action_space (int): how many dimensions the action space has
        action_idx (int): the index we are inserting a 1 on the resulting array

    Returns:
        None
    """
    z = torch.zeros(action_space)
    z[action_idx] = 1
    return z


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    """
    Utility for debugging

    Args:
        message:
        category:
        filename:
        lineno:
        file:
        line:

    Returns:
        None
    """
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def one_hot_from_logits(logits, dim=0):
    """
    Is this being used? Not sure if I recall correctly but I remember issues with it.

    Args:
        logits:
        dim:

    Returns:

    """
    return (logits == logits.max(1, keepdim=True)[dim]).float()


def terminate_process():
    """
    Function to pkill all processes that had been spawned by Shiva

    Returns:
        None
    """
    system = platform.system()
    if system == 'Darwin':
        cmd = "pkill -f 'Python shiva'"
    else:
        cmd = "pkill -e -f 'python shiva'"
    time.sleep(1)
    # subprocess.call(cmd, shell=True)


def flat_1d_list(list):
    """
    Converts a two dimenional python list into a one dimensional

    Args:
        list (List): two dimensional python list

    Returns:
        List: one dimensional list
    """
    return [item for sublist in list for item in sublist]
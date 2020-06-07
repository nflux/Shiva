import sys
import time
import traceback, warnings
import numpy as np
import torch
import subprocess
import platform

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
    '''
        Returns a one-hot encoded numpy.ndarray
    '''
    z = np.zeros(action_space)
    z[action_idx] = 1
    return z if numpy else list(z)

def action2one_hot_v(action_space: int, action_idx: int) -> torch.tensor:
    '''
        Returns a one-hot encoded torch.tensor
    '''
    z = torch.zeros(action_space)
    z[action_idx] = 1
    return z

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    '''
        Utility for debugging
        Comment last line to enable/disable
    '''
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def one_hot_from_logits(logits, dim=0):
    return (logits == logits.max(1, keepdim=True)[dim]).float()

def terminate_process():
    system = platform.system()
    if system == 'Darwin':
        cmd = "pkill -f 'Python shiva'"
    else:
        cmd = "pkill -e -f 'python shiva'"
    time.sleep(1)
    # subprocess.call(cmd, shell=True)

def flat_1d_list(list):
    return [item for sublist in list for item in sublist]
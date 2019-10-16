import numpy as np
import torch

def zero_params(x):
    if len(x.shape) == 3: # reshape sequence to be inside batch dimension
        seq = x.shape[0]
        batch = x.shape[1]
        x = x.reshape(seq*batch,8)
    if len(x.shape) ==2:
        dash = torch.tensor([1,0,0], device=x.device).float()
        dash = dash.repeat(len(x),1)
        turn = torch.tensor([0,1,0], device=x.device).float()
        turn = turn.repeat(len(x),1)
        kick = torch.tensor([0,0,1], device=x.device).float()
        kick = kick.repeat(len(x),1)
        index_dash = torch.nonzero((x[:,:3] == dash).sum(dim=1) == x[:,:3].size(1))
        index_turn = torch.nonzero((x[:,:3] == turn).sum(dim=1) == x[:,:3].size(1))
        index_kick = torch.nonzero((x[:,:3] == kick).sum(dim=1) == x[:,:3].size(1))   
        x[index_dash,5:] = 0.0
        x[index_turn,3:5] = 0.0
        x[index_turn,6:] =0.0
        x[index_kick,3:-2]=0.0
        return x.reshape(seq,batch,8)

    if len(x.shape) == 1: # Single numpy array (rc_env)
        if np.all(x[:3] == [1.0,0.0,0.0]):
            x[5:] = 0.0
        elif np.all(x[:3] == [0.0,1.0,0.0]):
            x[3:5] = 0.0
            x[6:] =0.0
        elif np.all(x[:3] == [0.0,0.0,1.0]):
            x[3:-2]=0.0
        else:
            print("Error action not one hot")
        return x
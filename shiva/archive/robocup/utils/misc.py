import math
import re
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import shutil
import time
import pandas as pd
import collections

# Flatten an N-dim irregular list to 1-Dim
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def prep_session(session_path="",hist_dir="history",eval_hist_dir= "eval_history",eval_log_dir = "eval_log",load_path = "models/",ensemble_path = "ensemble_models/",log_dir="logs",num_TA=1):
    hist_dir = hist_dir
    eval_hist_dir = eval_hist_dir
    eval_log_dir =  eval_log_dir
    load_path =  load_path
    ensemble_path = ensemble_path
    directories = [session_path,eval_log_dir, eval_hist_dir,load_path, hist_dir, log_dir,ensemble_path]

    [shutil.rmtree(path) for path in directories if os.path.isdir(path)]
    [shutil.rmtree(path) for path in [ensemble_path + ("ensemble_agent_%i" % j) for j in range(num_TA)] if os.path.isdir(path)] 
    [shutil.rmtree(path) for path in [load_path + ("agent_%i" % j) for j in range(num_TA)] if os.path.isdir(path)] 
    [shutil.rmtree(path) for path in [session_path + ("agent2d")] if os.path.isdir(path)] 

    [os.makedirs(path) for path in [ensemble_path + ("ensemble_agent_%i" % j) for j in range(num_TA)] if not os.path.exists(path)] # generate ensemble model paths for each agent
    [os.makedirs(path) for path in [load_path + ("agent_%i" % j) for j in range(num_TA)] if not os.path.exists(path)] # generate model paths for each agent
    [os.makedirs(path) for path in [session_path + ("agent2d")] if not os.path.exists(path)] # generate model paths for each agent

    [os.makedirs(path) for path in directories if not os.path.exists(path)] # generate directories 
    
    

def processor(tensor,device,torch_device="cuda:0"):
    if device == 'cuda':
        fn = lambda x: x.to(torch_device)
    else:
        fn = lambda x: x.cpu()
    return fn(tensor)


def e_greedy(logits, numAgents, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    
    ** Modified to return True if random action is taken, else return False
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs, [False] * numAgents
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    # explore = False
    ex_list = [False] * numAgents
    rand = torch.rand(logits.shape[0])
    for i,r in enumerate(rand):
        if r < eps:
            ex_list[i] = True
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(rand)]) , ex_list

def e_greedy_bool(numAgents, eps=0.0,device='cpu'):
    """

    
    Return True if random action is should be takent (determined by e-greedy), else return False
    """
    # get best (according to current policy) actions in one-hot form
    if eps == 0.0:
        return torch.zeros(numAgents,device=device,requires_grad=False)
    # get random actions in one-hot form
    # chooses between best and random actions using epsilon greedy
    # explore = False
    eps_list = torch.full((1,numAgents),eps)
    rand = torch.empty(numAgents,device=device,requires_grad=False).uniform_(0,1)
    return (rand < eps)


def pretrain_process(left_fnames, right_fnames, fstatus, num_features):
    
    # sort fnames by uniform
    left_fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    right_fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    obs_header_names = ['cycle']
    for n in range(num_features):
        obs_header_names.append(str(n))
    print("Reading CSVs")
    df_left_action_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'dash', 'turn', 'kick', 'd1', 'd2', 't1', 'k1', 'k2']) for fn in left_fnames if '_actions_left' in fn]
    df_left_obs_list = [pd.read_csv(fn, sep=',', header=None, names=obs_header_names) for fn in left_fnames if '_obs_left' in fn]

    #df_right_status_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'status']) for fn in right_fnames if '_status_' in fn]
    df_right_action_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'dash', 'turn', 'kick', 'd1', 'd2', 't1', 'k1', 'k2']) for fn in right_fnames if '_actions_right' in fn]
    df_right_obs_list = [pd.read_csv(fn, sep=',', header=None, names=obs_header_names) for fn in right_fnames if '_obs_right' in fn]

    status = pd.read_csv(fstatus, sep=',', header=None, names=['cycle', 'status'])
    # print(df_left_action_list)
    print("Dropping duplicates")
    # Drop repeated cycles
    status.drop_duplicates(['cycle'], keep='last', inplace=True)
    [df.drop_duplicates(['cycle'], keep=False, inplace=True) for df in df_left_action_list]
    [df.drop_duplicates(['cycle'], keep='last', inplace=True) for df in df_left_obs_list]

    [df.drop_duplicates(['cycle'], keep=False, inplace=True) for df in df_right_action_list]
    [df.drop_duplicates(['cycle'], keep='last', inplace=True) for df in df_right_obs_list]

    # # Temporary fix for messed up goalie actions
    df_goalie_action_patch = pd.DataFrame()
    df_goalie_action_patch['cycle'] = np.arange(len(status.index)+1)
    print("Cleaning null actions")
    df_left_action_list = [pd.merge(df_goalie_action_patch, df_left_action_list[i], on='cycle', how='outer') for i in range(len(df_left_action_list))]
    df_right_action_list = [pd.merge(df_goalie_action_patch, df_right_action_list[i], on='cycle', how='outer') for i in range(len(df_right_action_list))]
    #[df.interpolate(inplace=True) for df in df_left_action_list]
    #[df.interpolate(inplace=True) for df in df_right_action_list]
    [df.fillna(0.0,inplace=True) for df in df_left_action_list]
    [df.fillna(0.0,inplace=True) for df in df_right_action_list]
    print("Checking DF's for null values")
    print([df.isnull().values.any() for df in df_left_action_list])
    print([df.isnull().values.any() for df in df_right_action_list])

    
    df_goalie_action_patch = pd.DataFrame()
    df_goalie_action_patch['cycle'] = np.arange(len(status.index)+1)
    print("Cleaning null obs")
    df_left_obs_list = [pd.merge(df_goalie_action_patch, df_left_obs_list[i], on='cycle', how='outer') for i in range(len(df_left_obs_list))]
    df_right_obs_list = [pd.merge(df_goalie_action_patch, df_right_obs_list[i], on='cycle', how='outer') for i in range(len(df_right_obs_list))]
    #[df.interpolate(inplace=True) for df in df_left_obs_list]
    #[df.interpolate(inplace=True) for df in df_right_obs_list]
    [df.fillna(0.0,inplace=True) for df in df_left_obs_list]
    [df.fillna(0.0,inplace=True) for df in df_right_obs_list]
    print("Checking DF's for null values")


    
    df_goalie_action_patch = pd.DataFrame()
    df_goalie_action_patch['cycle'] = np.arange(len(status.index))
    print("Cleaning null status")
    status = pd.merge(df_goalie_action_patch, status, on='cycle', how='outer')
    status.fillna(0.0,inplace=True)
    # [df.fillna(0,inplace=True) for df in status_list]
    #[df.fillna(0,inplace=True) for df in df_right_status_list]
    print("Checking DF's for null values")
    # print([df.isnull().values.any() for df in status_list])
    #print([df.isnull().values.any() for df in df_right_status_list])

    # [df.interpolate() for df in df_right_action_list]
    # [df.interpolate() for df in df_left_action_list]
    # df_left_action_list[0].to_csv('./temp_path_robocup0.csv', sep=',', index=False)
    # df_left_action_list[1].to_csv('./temp_path_robocup1.csv', sep=',', index=False)
    # df_left_action_list[2].to_csv('./temp_path_robocup2.csv', sep=',', index=False)

    # df_right_action_list[0].to_csv('./temp_path_robocup3.csv', sep=',', index=False)
    # df_right_action_list[1].to_csv('./temp_path_robocup4.csv', sep=',', index=False)
    # df_right_action_list[2].to_csv('./temp_path_robocup5.csv', sep=',', index=False)
    # exit(0)
    print("Converting df to numpy")
    # Turn to numpy arrays, NOTE: Actions are hot-encoded in the logs
    status = status.loc[:, 'status'].values
    team_pt_obs = [df.loc[:, obs_header_names[1:]].values for df in df_left_obs_list]
    team_pt_actions = [df.loc[:, 'dash':].values for df in df_left_action_list]

    #o_status = [df.loc[:, 'status'].values for df in df_right_status_list]
    opp_pt_obs = [df.loc[:, obs_header_names[1:]].values for df in df_right_obs_list]
    opp_pt_actions = [df.loc[:, 'dash':].values for df in df_right_action_list]
    

    # Change dims to timesteps x number of agents x item
    #team_pt_status = [np.asarray([status[i][ts] for i in range(len(status))]) for ts in range(len(status[0])-3)]
    team_pt_obs = [np.asarray([team_pt_obs[i][ts] for i in range(len(team_pt_obs))]) for ts in range(len(status)-3)]
    team_pt_actions = [np.asarray([team_pt_actions[i][ts] for i in range(len(team_pt_actions))]) for ts in range(len(status)-3)]

    #opp_pt_status = [np.asarray([o_status[i][ts] for i in range(len(o_status))]) for ts in range(len(status[0])-3)]
    opp_pt_obs = [np.asarray([opp_pt_obs[i][ts] for i in range(len(opp_pt_obs))]) for ts in range(len(status)-3)]
    opp_pt_actions = [np.asarray([opp_pt_actions[i][ts] for i in range(len(opp_pt_actions))]) for ts in range(len(status)-3)]
    team_pt_status = 0
    opp_pt_status = 0
    
    return team_pt_status, team_pt_obs, team_pt_actions, opp_pt_status, opp_pt_obs, opp_pt_actions, status

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
# returns the distribution projection
def distr_projection(self,next_distr_v, rewards_v, dones_mask_t, cum_rewards_v, gamma, device="cpu"):
    start = time.time()
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    cum_rewards = cum_rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, self.N_ATOMS), dtype=np.float32)
    mask = [True]*batch_size

    start=time.time()
    for atom in range(self.N_ATOMS):
        tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, 
                                                (1-self.beta)*(rewards + (self.Vmin + atom * self.DELTA_Z) * gamma) + (self.beta)*cum_rewards))
        b_j = (tz_j - self.Vmin) / self.DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)

        #eq_mask = u == l
        #proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        #ne_mask = u != l
        proj_distr[mask, l[mask]] += next_distr[mask, atom] * (u - b_j)[mask]
        proj_distr[mask, u[mask]] += next_distr[mask, atom] * (b_j - l)[mask]


    if dones_mask.any():
        proj_distr[dones_mask] = 0.0

        tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[dones_mask]))
        b_j = (tz_j - self.Vmin) / self.DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    #print(time.time() - start,"TIME FOR PROJ")
    #print(time.time() - start,"time1")
    #proj_distr2=proj_distr
    #return torch.from_numpy(proj_distr).float().to(device)
# ---------------------------------------------------------------
#start = time.time()
#     next_distr = next_distr_v.data.cpu().numpy()
#     rewards = rewards_v.data.cpu().numpy()
#     cum_rewards = cum_rewards_v.data.cpu().numpy()
#     dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
#     batch_size = len(rewards)
#     proj_distr = np.zeros((batch_size, self.N_ATOMS), dtype=np.float32)
#     mask = [True]*batch_size
#     start = time.time()
#     atoms = np.arange(self.N_ATOMS,dtype='float64') 
#     atoms *= self.DELTA_Z
#     atoms += self.Vmin
#     atoms *= gamma
#     atoms = np.tile(atoms,(self.batch_size,1))

#     atoms += np.tile((rewards),(self.N_ATOMS,1)).T
#     atoms *= (1-self.beta)
#     atoms += np.tile((self.beta*cum_rewards),(self.N_ATOMS,1)).T
#     atoms[atoms > self.Vmax] = self.Vmax
#     atoms[atoms < self.Vmin] =  self.Vmin
#     b_j = (atoms - self.Vmin) / self.DELTA_Z
#     l = np.floor(b_j).astype(np.int64)
#     u = l+1
#     #u = u.T
    
#     proj_distr = proj_distr.T
#     #next_distr = next_distr.T

#     for i in range(self.N_ATOMS):
#         proj_distr[l[:,i],mask] += next_distr[mask,i]*(u-b_j)[mask,i]
#         proj_distr[u[:,i],mask] += next_distr[mask,i]*(b_j-l)[mask,i]
#     proj_distr = proj_distr.T
#     #next_distr = next_distr.T
#     print(time.time() - start,"time2")

#     if dones_mask.any():
#         proj_distr[dones_mask] = 0.0

#         tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[dones_mask]))
#         b_j = (tz_j - self.Vmin) / self.DELTA_Z
#         l = np.floor(b_j).astype(np.int64)
#         u = np.ceil(b_j).astype(np.int64)
#         #eq_mask = u == l
#         eq_dones = dones_mask.copy()
#         eq_dones[dones_mask] = eq_mask
#         if eq_dones.any():
#             proj_distr[eq_dones, l[eq_mask]] = 1.0
#         ne_mask = u != l
#         ne_dones = dones_mask.copy()
#         ne_dones[dones_mask] = ne_mask
#         if ne_dones.any():
#             proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
#             proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.from_numpy(proj_distr).float().to(device)

# ---------------- Torch Tensorized ------------------------------
#     with torch.no_grad():
#         start = time.time()

#         device='cpu'
#         next_distr = next_distr_v.to(device)
#         rewards = rewards_v.to(device)
#         cum_rewards = cum_rewards_v.to(device)


#         dones_mask = dones_mask_t.byte().to(device)
#         batch_size = len(rewards)
#         proj_distr = torch.zeros(batch_size, self.N_ATOMS,device=device)

#         tensor_Vmin = torch.ones(batch_size).to(device) * self.Vmin
#         tensor_Vmax = torch.ones(batch_size).to(device) * self.Vmax

#         eq_mask = torch.ByteTensor(batch_size)
#         ne_mask = torch.ByteTensor(batch_size)


#         # If we can parallize the atoms into matrices form maybe we can get an improvement
#         for atom in range(self.N_ATOMS):

#             tz_j = torch.min(tensor_Vmax, torch.max(tensor_Vmin, 
#                                                     (1-self.beta)*(rewards + (self.Vmin + atom * self.DELTA_Z) * gamma) + (self.beta)*cum_rewards))
#             b_j = torch.add(tz_j, - self.Vmin) / self.DELTA_Z

#             l = torch.floor(b_j).long()
#             u = torch.ceil(b_j).long()


#             eq_mask = (u == l)
#             ne_mask = (u != l)
#             #start = time.time()


#             #end = time.time()
#             #print(end-start)
#             proj_distr[eq_mask,l[eq_mask]] += next_distr[eq_mask,atom]

#             proj_distr[ne_mask,l[ne_mask]] += next_distr[ne_mask,atom] * (u.float() - b_j)[ne_mask]
#             proj_distr[ne_mask,u[ne_mask]] += next_distr[ne_mask,atom] * (b_j - l.float())[ne_mask]

#         if dones_mask.any():
#             proj_distr[dones_mask] = 0.0
#             tensor_Vmin = torch.ones_like(rewards[dones_mask]) * self.Vmin
#             tensor_Vmax = torch.ones_like(rewards[dones_mask]) * self.Vmax

#             tz_j = torch.min(tensor_Vmax, torch.max(tensor_Vmin, rewards[dones_mask]))
#             b_j =  torch.add(tz_j, - self.Vmin) / self.DELTA_Z
#             l = torch.floor(b_j).long()
#             u = torch.ceil(b_j).long()
#             eq_mask = (u == l)
#             eq_dones = dones_mask.clone()
#             eq_dones[dones_mask] = eq_mask
#             if eq_dones.any():
#                 proj_distr[eq_dones, l[eq_mask]] = 1.0
#             ne_mask = (u != l)
#             ne_dones = dones_mask.clone()
#             ne_dones[dones_mask] = ne_mask
#             if ne_dones.any():
#                 proj_distr[ne_dones, l[ne_mask]] = (u.float() - b_j)[ne_mask]
#                 proj_distr[ne_dones, u[ne_mask]] = (b_j - l.float())[ne_mask]
#     print(time.time() - start,"TIME FOR PROJ")

#     return proj_distr.to('cuda')
# # ------------------------------------------------------


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

# https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def onehot_from_logits(logits, eps=0.0,LSTM=False):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    if not LSTM:
        argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    else:
        argmax_acs = (logits == logits.max(2, keepdim=True)[0]).float()

    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape,eps=1e-20,tens_type=torch.FloatTensor,device="cuda",LSTM=False):  
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature,device="cuda",LSTM=False):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data),device=device,LSTM=LSTM)
    if LSTM:
        return F.softmax(y/temperature, dim=2)
    else:
        return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False,device="cuda",LSTM=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized logs-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature,device,LSTM=LSTM)
    if hard:
        y_hard = onehot_from_logits(y,LSTM=LSTM)
        y = (y_hard - y).detach() + y
    return y
    

def getPretrainRew(s,d,base):


    reward=0.0
    team_reward = 0.0
    goal_points = 30.0
    #---------------------------
    if d:
        if 'base_left' == base:
        # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----
            if s==1:
                reward+= goal_points
            elif s==2:
                reward+=-goal_points
            elif s==3:
                reward+=-0.5
            elif s==6:
                reward+= +goal_points/5.0
            elif s==7:
                reward+= -goal_points/4.0

            return reward
        else:
            if s==1:
                reward+=-goal_points
            elif s==2:
                reward+=goal_points
            elif s==3:
                reward+=-0.5
            elif s==6:
                reward+= -goal_points/4.0
            elif s==7:
                reward+= goal_points/5.0

    return reward

def exp_stack(obs,acs,rews,next_obs,dones,MC_targets,n_step_targets,ws,prio,def_prio,LSTM_policy):
    if not LSTM_policy:
        return np.column_stack((obs, acs, rews, next_obs, dones, MC_targets, n_step_targets,
                                    ws, prio, def_prio))
    else:
        return np.column_stack((obs, acs, rews, dones, MC_targets, n_step_targets,
                                    ws, prio, def_prio))

# Handles 
def push(team_replay_buffer,opp_replay_buffer,num_envs,shared_exps,exp_indices,num_TA,ep_num,seq_length,LSTM,push_only_left):
    for i in range(num_envs):    
        if LSTM:
            if push_only_left:
                [team_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], :num_TA, :]) for j in range(int(ep_num[i].item())) if seq_length-1 <= exp_indices[i][j]]
                [opp_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], num_TA:2*num_TA, :]) for j in range(int(ep_num[i].item())) if seq_length-1 <= exp_indices[i][j]]
            else:
                [team_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], :num_TA, :], 
                    shared_exps[i][j][:exp_indices[i][j], -num_TA:, :]))) for j in range(int(ep_num[i].item())) if seq_length-1 <= exp_indices[i][j]]
                [opp_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], -num_TA:, :], 
                    shared_exps[i][j][:exp_indices[i][j], :num_TA, :]))) for j in range(int(ep_num[i].item())) if seq_length-1 <= exp_indices[i][j]]
        else:
            if push_only_left:
                [team_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], :num_TA, :]) for j in range(int(ep_num[i].item()))]
                [opp_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], num_TA:2*num_TA, :]) for j in range(int(ep_num[i].item()))]
            else:
                [team_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], :num_TA, :], shared_exps[i][j][:exp_indices[i][j], -num_TA:, :]))) for j in range(int(ep_num[i].item()))]
                [opp_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], -num_TA:, :], shared_exps[i][j][:exp_indices[i][j], :num_TA, :]))) for j in range(int(ep_num[i].item()))]

# NOTE: Assumes agents are loaded in sorted uniform order. Ergo 1,2,3,...
def pt_constructProxmityList(all_tobs, all_oobs, all_tacs, all_oacs, num_agents):
    sortedByProxRet = []
    sortedUnumOurList = None
    sortedUnumOppList = None
    for i in range(num_agents):
        agent_exp = []
        sortedUnumOurList, sortedUnumOppList = pt_distances(num_agents, i, all_tobs, all_oobs)
        agent_exp.append(all_tobs[sortedUnumOurList])
        agent_exp.append(all_oobs[sortedUnumOppList])
        agent_exp.append(all_tacs[sortedUnumOurList])
        agent_exp.append(all_oacs[sortedUnumOppList])

        sortedByProxRet.append(agent_exp)

    return sortedByProxRet

# NOTE: Assumes agents are loaded in sorted uniform order. Ergo 1,2,3,...
def constructProxmityList(env, all_tobs, all_oobs, all_tacs, all_oacs, num_agents, side):
    sortedByProxRet = []
    sortedUnumOurList = None
    sortedUnumOppList = None
    for i in range(num_agents):
        agent_exp = []
        sortedUnumOurList, sortedUnumOppList = env.distances(i, side)
        agent_exp.append(all_tobs[sortedUnumOurList])
        agent_exp.append(all_oobs[sortedUnumOppList])
        agent_exp.append(all_tacs[sortedUnumOurList])
        agent_exp.append(all_oacs[sortedUnumOppList])
        sortedByProxRet.append(agent_exp)

    return sortedByProxRet

def convertProxListToTensor(all_prox_lists, agents, item_size):
    num_steps = len(all_prox_lists)
    prox_tensor = torch.zeros(num_steps, agents, item_size)
    for i,apl in enumerate(all_prox_lists):
        gen = flatten(apl)
        np_list = np.array(list(gen))
        for a in range(agents):
            prox_tensor[i,a] = torch.from_numpy(np_list[a*item_size:item_size*(a+1)])
    
    return prox_tensor

def distance(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def pt_distances(num_agents, agentID, tobs, oobs):
    x = 20
    y = 21
    distances_team = []
    distances_opp = []
    for i in range(num_agents):
        distances_team.append(distance(tobs[agentID][x],tobs[i][x], tobs[agentID][y],tobs[i][y]))
        distances_opp.append(distance(tobs[agentID][x], -oobs[i][x], tobs[agentID][y], -oobs[i][y]))
    return np.argsort(distances_team), np.argsort(distances_opp)

def distance(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def load_buffer(left,right,fstatus,zip_vars):
    num_TA,obs_dim_TA,acs_dim,team_PT_replay_buffer,opp_PT_replay_buffer,episode_length,n_steps,gamma,D4PG,SIL,k_ensembles,push_only_left,LSTM_policy,prox_item_size = zip_vars
    
    
    team_pt_status, team_pt_obs,team_pt_actions, opp_pt_status, opp_pt_obs, opp_pt_actions, status = pretrain_process(left_fnames=left, right_fnames=right, fstatus=fstatus, num_features = obs_dim_TA-acs_dim) # -8 for action space thats added here.

    # Count up everything besides IN_GAME to get number of episodes
    collect = collections.Counter(status)
    pt_episodes = collect[1] + collect[2] + collect[3] + collect[4] + collect[6] + collect[7]
    
    pt_time_step = 0
    critic_mod_both = True
    num_OA = num_TA
    ################## Base Left #########################
    for ep_i in range(pt_episodes-10):
        if ep_i % 100 == 0:
            print("Pushing Pretrain Episode:",ep_i)

        team_n_step_rewards = []
        team_n_step_obs = []
        team_n_step_acs = []
        team_n_step_next_obs = []
        team_n_step_dones = []
        team_n_step_ws = []

        opp_n_step_rewards = []
        opp_n_step_obs = []
        opp_n_step_acs = []
        opp_n_step_next_obs = []
        opp_n_step_dones = []
        opp_n_step_ws = []
        
        # List of tensors sorted by proximity in terms of agents
        sortedByProxTeamList = []
        sortedByProxOppList = []
        d = 0
        first_action = np.asarray([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] for _ in range(num_TA)])
        for et_i in range(episode_length):
            exps = None
            world_stat = status[pt_time_step+1]
            d = 0
            if world_stat != 0.0:
                d = 1

            if et_i == 0:                
                obs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(team_pt_obs[pt_time_step],first_action)])
                opp_obs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(opp_pt_obs[pt_time_step],first_action)])
            else:
                obs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(team_pt_obs[pt_time_step],team_pt_actions[pt_time_step-1])])
                opp_obs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(opp_pt_obs[pt_time_step],opp_pt_actions[pt_time_step-1])])
            nobs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(team_pt_obs[pt_time_step+1],team_pt_actions[pt_time_step])])
            opp_nobs = np.asarray([list(np.concatenate((t,a),axis=0)) for t,a in zip(opp_pt_obs[pt_time_step+1],opp_pt_actions[pt_time_step])])

            sortedByProxTeamList.append(pt_constructProxmityList(obs, opp_obs, team_pt_actions[pt_time_step], opp_pt_actions[pt_time_step], num_TA))
            sortedByProxOppList.append(pt_constructProxmityList(opp_obs, obs, opp_pt_actions[pt_time_step], team_pt_actions[pt_time_step], num_OA))

            team_n_step_acs.append(team_pt_actions[pt_time_step]) 
            team_n_step_obs.append(obs)
            team_n_step_ws.append(world_stat)
            team_n_step_next_obs.append(nobs)
            team_n_step_rewards.append(np.hstack([getPretrainRew(world_stat,d,"base_left") for i in range(num_TA)]))          
            team_n_step_dones.append(d)
            
            opp_n_step_acs.append(opp_pt_actions[pt_time_step])
            opp_n_step_obs.append(opp_obs)
            opp_n_step_ws.append(world_stat)
            opp_n_step_next_obs.append(opp_nobs)
            opp_n_step_rewards.append(np.hstack([getPretrainRew(world_stat,d,"base_right") for i in range(num_TA)]))          
            opp_n_step_dones.append(d)

            # Store variables for calculation of MC and n-step targets for team
            pt_time_step += 1
            if d == 1: # Episode done
                pt_time_step += 1 # Jump ahead 1 if episode over
                n_step_gammas = np.array([[gamma**step for a in range(num_TA)] for step in range(n_steps)])
                # NOTE: Assume M vs M and critic_mod_both == True
                if critic_mod_both:
                    team_all_MC_targets = []
                    opp_all_MC_targets = []
                    MC_targets_team = np.zeros(num_TA)
                    MC_targets_opp = np.zeros(num_OA)
                    for n in range(et_i+1):
                        MC_targets_team = team_n_step_rewards[et_i-n] + MC_targets_team*gamma
                        team_all_MC_targets.append(MC_targets_team)
                        MC_targets_opp = opp_n_step_rewards[et_i-n] + MC_targets_opp*gamma
                        opp_all_MC_targets.append(MC_targets_opp)
                    for n in range(et_i+1):
                        n_step_targets_team = np.zeros(num_TA)
                        n_step_targets_opp = np.zeros(num_OA)
                        if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                            n_step_targets_team = np.sum(np.multiply(np.asarray(team_n_step_rewards[n:n+n_steps]),(n_step_gammas)),axis=0)
                            n_step_targets_opp = np.sum(np.multiply(np.asarray(opp_n_step_rewards[n:n+n_steps]),(n_step_gammas)),axis=0)

                            n_step_next_ob_team = team_n_step_next_obs[n - 1 + n_steps]
                            n_step_done_team = team_n_step_dones[n - 1 + n_steps]

                            n_step_next_ob_opp = opp_n_step_next_obs[n - 1 + n_steps]
                            n_step_done_opp = opp_n_step_dones[n - 1 + n_steps]
                        else: # n-step = MC if less than n steps remaining
                            n_step_targets_team = team_all_MC_targets[et_i-n]
                            n_step_next_ob_team = team_n_step_next_obs[-1]
                            n_step_done_team = team_n_step_dones[-1]

                            n_step_targets_opp = opp_all_MC_targets[et_i-n]
                            n_step_next_ob_opp = opp_n_step_next_obs[-1]
                            n_step_done_opp = opp_n_step_dones[-1]
                        if D4PG:
                            default_prio = 5.0
                        else:
                            default_prio = 3.0
                        current_ensembles = [0]*num_TA
                        priorities = np.array([np.zeros(k_ensembles) for i in range(num_TA)])
                        priorities[:,current_ensembles] = 5.0
                        # print(current_ensembles)
                        if SIL:
                            SIL_priorities = np.ones(num_TA)*default_prio
                        
                        # If LSTM_policy == True then don't take the next obs
                        exp_team = exp_stack(team_n_step_obs[n],
                                            team_n_step_acs[n],
                                            np.expand_dims(team_n_step_rewards[n], 1),
                                            n_step_next_ob_team,
                                            np.expand_dims([n_step_done_team for i in range(num_TA)], 1),
                                            np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_team, 1),
                                            np.expand_dims([team_n_step_ws[n] for i in range(num_TA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(num_TA)],1), LSTM_policy)

                        exp_opp = exp_stack(opp_n_step_obs[n],
                                            opp_n_step_acs[n],
                                            np.expand_dims(opp_n_step_rewards[n], 1),
                                            n_step_next_ob_opp,
                                            np.expand_dims([n_step_done_opp for i in range(num_OA)], 1),
                                            np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_opp, 1),
                                            np.expand_dims([opp_n_step_ws[n] for i in range(num_OA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(num_TA)],1), LSTM_policy)
                
                        exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)
                        
                        if exps is None:
                            exps = torch.from_numpy(exp_comb)
                        else:
                            exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)
                        
                    prox_team_tensor = convertProxListToTensor(sortedByProxTeamList, num_TA, prox_item_size)
                    prox_opp_tensor = convertProxListToTensor(sortedByProxOppList, num_OA, prox_item_size)
                    comb_prox_tensor = torch.cat((prox_team_tensor, prox_opp_tensor), dim=1)
                    # Fill in values for zeros for the list of list of lists
                    exps = torch.cat((exps[:, :, :], comb_prox_tensor.double()), dim=2)

                    if ep_i != 0:
                        
                        if not LSTM_policy:
                            if push_only_left:
                                team_PT_replay_buffer.push(exps[:, :num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                            else:
                                team_PT_replay_buffer.push(exps[:, :num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, :num_TA, :])
                                team_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                                #team_PT_replay_buffer.push(torch.cat((exps[:, :num_TA, :], exps[:,-num_TA:,:])))
                                #opp_PT_replay_buffer.push(torch.cat((exps[:, -num_TA:, :], exps[:,:num_TA,:])))
                        else:
                            if push_only_left:
                                team_PT_replay_buffer.push(exps[:, :num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                            else:
                                team_PT_replay_buffer.push(exps[:, :num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                                opp_PT_replay_buffer.push(exps[:, :num_TA, :])
                                team_PT_replay_buffer.push(exps[:, num_TA:2*num_TA, :])
                                #team_PT_replay_buffer.push(torch.cat((exps[:, :num_TA, :], exps[:,-num_TA:,:])))
                                #opp_PT_replay_buffer.push(torch.cat((exps[:, -num_TA:, :], exps[:,:num_TA,:])))
        
                    del exps
                    exps = None
                break
    del team_pt_obs
    del team_pt_status
    del team_pt_actions

    del opp_pt_obs
    del opp_pt_status
    del opp_pt_actions
    return (team_PT_replay_buffer,opp_PT_replay_buffer)
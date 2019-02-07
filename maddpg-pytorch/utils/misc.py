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

def prep_session(session_path="",hist_dir="history",eval_hist_dir= "eval_history",eval_log_dir = "eval_log",load_path = "models/",ensemble_path = "ensemble_models/",log_dir="log",num_TA=1):
    hist_dir = hist_dir
    eval_hist_dir = eval_hist_dir
    eval_log_dir =  eval_log_dir
    load_path =  load_path
    ensemble_path = ensemble_path
    directories = [session_path,eval_log_dir, eval_hist_dir,load_path, hist_dir, log_dir,ensemble_path]

    [shutil.rmtree(path) for path in directories if os.path.isdir(path)]
    [shutil.rmtree(path) for path in [ensemble_path + ("ensemble_agent_%i" % j) for j in range(num_TA)] if os.path.isdir(path)] 
    [shutil.rmtree(path) for path in [load_path + ("agent_%i" % j) for j in range(num_TA)] if os.path.isdir(path)] 
    [os.makedirs(path) for path in [ensemble_path + ("ensemble_agent_%i" % j) for j in range(num_TA)] if not os.path.exists(path)] # generate ensemble model paths for each agent
    [os.makedirs(path) for path in [load_path + ("agent_%i" % j) for j in range(num_TA)] if not os.path.exists(path)] # generate model paths for each agent
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


def pretrain_process(left_fnames, right_fnames, timesteps, num_features):
    
    # sort fnames
    left_fnames.sort()
    right_fnames.sort()

    obs_header_names = ['cycle', 'item']
    for n in range(num_features):
        obs_header_names.append(str(n))

    df_left_status_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'item', 'status']) for fn in left_fnames if '_status_' in fn]
    df_left_action_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'action', 'param1', 'param2']) for fn in left_fnames if '_actions_' in fn]
    df_left_obs_list = [pd.read_csv(fn, sep=',', header=None, names=obs_header_names) for fn in left_fnames if '_obs_' in fn]

    df_right_status_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'item', 'status']) for fn in right_fnames if '_status_' in fn]
    df_right_action_list = [pd.read_csv(fn, sep=',', header=None, names=['cycle', 'action', 'param1', 'param2']) for fn in right_fnames if '_actions_' in fn]
    df_right_obs_list = [pd.read_csv(fn, sep=',', header=None, names=obs_header_names) for fn in right_fnames if '_obs_' in fn]
    
    team_pt_status = [df.loc[:, 'status'].values for df in df_left_status_list]
    team_pt_obs = [df.loc[:, obs_header_names[2:]].values for df in df_left_obs_list]
    team_pt_actions = [df.loc[:, ]]
    opp_pt_status = []
    opp_pt_obs = []
    opp_pt_actions = []
    
    num_TA = len(fnames)/2
    c = 0

    print("Loading pretrain data")
    while c < (timesteps*3):
        pass
        
    # return team_pt_obs,team_pt_status,team_pt_actions, opp_pt_obs,opp_pt_status,opp_pt_actions
    return 0, 0, 0, 0, 0, 0



def zero_params(num_Agents,params,action_index):
    for i in range(num_Agents):
        if action_index[i] == 0:
            params[i][2] = 0
            params[i][3] = 0
            params[i][4] = 0
        if action_index[i] == 1:
            params[i][0] = 0
            params[i][1] = 0
            params[i][3] = 0
            params[i][4] = 0
        if action_index[i] == 2:
            params[i][0] = 0
            params[i][1] = 0
            params[i][2] = 0
    return params



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

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape,eps=1e-20,tens_type=torch.FloatTensor,device="cuda"):  
    """Sample from Gumbel(0, 1)"""
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature,device="cuda"):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data),device=device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False,device="cuda"):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature,device)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y

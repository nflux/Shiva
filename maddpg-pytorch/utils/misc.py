import re
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import numpy as np
import shutil

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
    
    

def processor(tensor,device):
    if device == 'cuda':
        fn = lambda x: x.cuda()
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

def e_greedy_bool(numAgents, eps=0.0):
    """

    
    Return True if random action is should be takent (determined by e-greedy), else return False
    """
    # get best (according to current policy) actions in one-hot form
    if eps == 0.0:
        return np.zeros(numAgents)
    # get random actions in one-hot form
    # chooses between best and random actions using epsilon greedy
    # explore = False
    eps_list = np.full(numAgents,eps)
    rand = np.random.uniform(0,1,numAgents)
    return (rand < eps)


def pretrain_process(fnames,timesteps,num_features):
    contents = []
    for fname in fnames:
        print(fname)

        with open(fname) as f:
            contents.append(f.readlines())
                          
    for i in range(len(fnames)):
        contents[i] = [x.strip() for x in contents[i]]

        
    team_pt_status = []
    team_pt_obs = []
    team_pt_actions = []
    opp_pt_status = []
    opp_pt_obs = []
    opp_pt_actions = []
    
    num_TA = len(fnames)/2
    Tackle = False
    team_counters = np.array([0]*len(fnames))
    team_tackles = np.array([False]*len(fnames))
    screwy_tackle_counter = 0
    c = 0

    use_garbage_action = np.array([0]*len(fnames))
    print("Loading pretrain data")
    while c < (timesteps*3):
        if c % 30000 == 0:
            print("reading line",c/3)
        #print(c)
        #print(contents[0][team_counters[0]])
        #print(contents[1][team_counters[1]])
        #print(contents[2][team_counters[2]])
        #print(contents[3][team_counters[3]])
        #print(contents[4][team_counters[4]])
        #print(contents[5][team_counters[5]])
        screwy_tackle_counter = 0
        if contents[0][team_counters[0]].split(' ' )[3] == 'StateFeatures':
            team_obs = []
            opp_obs = []
            for agent in range(len(fnames)):
                ob = []
                for j in range(num_features):
                    ob.append(float(contents[agent][team_counters[agent]].split(' ')[4+j]))
                if agent < num_TA:
                    team_obs.append(ob)
                else:
                    opp_obs.append(ob)
            Tackle = False
            for agent in range(len(fnames)): #  handle no action in line
                if contents[agent][team_counters[agent] + 1].split(' ')[3] == 'GameStatus':
                    use_garbage_action[agent] = True
                elif contents[agent][team_counters[agent] + 1].split(' ')[3] == 'StateFeatures':
                    print("double state")
        elif 'agent' in contents[0][team_counters[0]].split(' ')[3]:
            for agent in range(len(fnames)):
                while "agent" in contents[agent][team_counters[agent]+1].split(' ')[3]:
                    action_string = contents[agent][team_counters[agent]+1].split(' ')[4] # If double Turn- Error
                    team_counters[agent] += 1 # Skip index for that agent
                if not contents[agent][team_counters[agent] + 1].split(' ')[3] == 'GameStatus':
                    print("error after action")
            team_all_as = []
            opp_all_as = []
            for agent in range(len(fnames)):
                action_string = contents[agent][team_counters[agent]].split(' ')[4]
                if use_garbage_action[agent]: # if action is missing for this timestep push 0's as action and do not consume 
                    # that line by += -1
                    a = np.zeros(8)
                    use_garbage_action[agent] = False
                    team_counters[agent] += -1
                elif "Dash"  in action_string: 
                    result = re.search('Dash((.*),*)', action_string)
                    power = float(result.group(1).split(',')[0][1:])
                    direction = float(result.group(1).split(',')[1][:-1])
                    #a = np.random.uniform(-1,1,8)
                    a = np.zeros(8)
                    a[0] += 1.0  
                    a[1] += 0.0 
                    a[2] += 0.0 
                    a[3] = power/100.0
                    a[4] = direction/180.0
                    a[5] = np.random.uniform(-.2,.2,1)
                    a[6] = np.random.uniform(-.2,.2,1)
                    a[7] = np.random.uniform(-.2,.2,1)

                elif "Turn"  in action_string:
                    result = re.search('Turn((.*))', action_string)
                    direction =float(result.group(1)[1:-1])
                    power = float(-1440)
                    a = np.zeros(8)
                    a[0] += 0.0  
                    a[1] += 1.0 
                    a[2] += 0.0 
                    a[3] = np.random.uniform(-.2,.2,1)
                    a[4] = np.random.uniform(-.2,.2,1)
                    a[5] = direction/180.0
                    a[6] = np.random.uniform(-.2,.2,1)
                    a[7] = np.random.uniform(-.2,.2,1)
                elif "Kick"  in action_string: 
                    result = re.search('Kick((.*),*)', action_string)
                    power = float(result.group(1).split(',')[0][1:])
                    direction = float(result.group(1).split(',')[1][:-1])
                    a = np.zeros(8)
                    a[0] += 0.0  
                    a[1] += 0.0 
                    a[2] += 1.0 
                    a[3] = np.random.uniform(-.2,.2,1)
                    a[4] = np.random.uniform(-.2,.2,1)
                    a[5] = np.random.uniform(-.2,.2,1)
                    a[6] = (power/100.0)*2 - 1
                    a[7] = direction/180.0
                elif "Tackle"  in action_string: 
                    result = re.search('Tackle((.*),*)', action_string)
                    power = float(result.group(1).split(',')[0][1:])
                    direction = float(result.group(1).split(',')[1][:-1])
                    # Throw away entry
                    Tackle = True
                    team_tackles[agent] = True # turned off for now ^
                else: # catch?
                    print("catch?")
                    a = np.random.uniform(-0.01,0.01,8)
                if agent < num_TA:
                    team_all_as.append(a)
                else:
                    opp_all_as.append(a)
        elif contents[0][team_counters[0]].split(' ')[3] == 'GameStatus':
            stat = float(contents[0][team_counters[0]].split(' ' )[4])
            if not contents[0][team_counters[0]+1].split(' ')[3] == 'StateFeatures':
                print("error after GS")
            if not Tackle:
                team_pt_actions.append([[x for x in ac] for ac in team_all_as])
                team_pt_obs.append(team_obs)
                team_pt_status.append(stat)
                opp_pt_actions.append([[x for x in ac] for ac in opp_all_as])
                opp_pt_obs.append(opp_obs)
                opp_pt_status.append(stat)
            else:
                tackler = np.where(team_tackles)[0][0] # convert to number from bool
                team_tackles = np.array([False]*len(fnames))
                #### Screwy offsets for tackle stun ####
                if not "agent" in contents[tackler][team_counters[tackler]+2].split(' ')[3]:
                    while not "agent" in contents[tackler][team_counters[tackler]].split(' ')[3]:
                        screwy_tackle_counter +=1
                        team_counters[tackler] +=1
                    screwy_tackle_counter += -2
                    team_counters[tackler] += -2
                    for ag in range(len(fnames)):
                        counter = 0
                        additional_counter = 0
                        while counter < (screwy_tackle_counter*3/2) + additional_counter:
                            if (ag != tackler):
                                while "agent" in contents[ag][team_counters[ag]+counter].split(' ')[3] and "agent" in contents[ag][team_counters[ag]+counter+1].split(' ')[3]:
                                    team_counters[ag] += 1
                                    additional_counter += 1
                            counter += 1
                                
                    team_counters += int((screwy_tackle_counter )* 3 / 2)
                    team_counters[tackler] += - int(screwy_tackle_counter *3/2) 
                    # agent who tackled
                
        team_counters += 1
        c += 1
    #team_pt_obs = np.asarray(np.asarray(pt_obs))
    team_pt_status = np.asarray(team_pt_status)
    return team_pt_obs,team_pt_status,team_pt_actions, opp_pt_obs,opp_pt_status,opp_pt_actions



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
    next_distr = next_distr_v.data.cpu().numpy()
    rewards = rewards_v.data.cpu().numpy()
    cum_rewards = cum_rewards_v.data.cpu().numpy()
    dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, self.N_ATOMS), dtype=np.float32)

    for atom in range(self.N_ATOMS):
        tz_j = np.minimum(self.Vmax, np.maximum(self.Vmin, 
                                                (1-self.beta)*(rewards + (self.Vmin + atom * self.DELTA_Z) * gamma) + (self.beta)*cum_rewards))
        b_j = (tz_j - self.Vmin) / self.DELTA_Z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]


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
    return torch.FloatTensor(proj_distr).to(device)


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

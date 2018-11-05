import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
import numpy as np
MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, a_lr=0.01, c_lr=0.01, hidden_dim=64,
                 discrete_action=True,vmax = 10,vmin = -10, N_ATOMS = 51, REWARD_STEPS = 5,
                 DELTA_Z = 20.0/50,D4PG=False,beta = 0):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,a_lr=a_lr, c_lr=c_lr,
                                 n_atoms = N_ATOMS, vmax = vmax, vmin = vmin,
                                 delta = DELTA_Z,D4PG=D4PG,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        self.REWARD_STEPS = REWARD_STEPS
        self.beta = beta
        self.N_ATOMS = N_ATOMS
        self.Vmax = vmax
        self.Vmin = vmin
        self.DELTA_Z = DELTA_Z
        self.D4PG = D4PG
    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]
    
    def scale_beta(self, beta):
        """
        Scale beta
        Inputs:
            scale (float): scale of beta
        """
        self.beta = beta

    
    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                       observations)]
    
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
                proj_distr[eq_dones, l] = 1.0
            ne_mask = u != l
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
        return torch.FloatTensor(proj_distr).to(device)
    
    # zeros the params corresponding to the non-chosen actions
    def zero_params(self,params,actions_oh):
        for a,p in zip(actions_oh,params):
            if np.argmax(a.data.numpy()) == 0:
                p[2 + len(a)] = 0 # offset by num of actions to get params
                p[3 + len(a)] = 0
                p[4 + len(a)] = 0
            if np.argmax(a.data.numpy()) == 1:
                p[0 + len(a)] = 0
                p[1 + len(a)] = 0
                p[3 + len(a)] = 0
                p[4 + len(a)] = 0
            if np.argmax(a.data.numpy()) == 2:
                p[0 + len(a)] = 0
                p[1 + len(a)] = 0
                p[2 + len(a)] = 0
        return params


    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks, cumulative discounted reward) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # rews = 1-step, cum-rews = n-step
        obs, acs, rews, next_obs, dones,cum_rews,nacs = sample
        curr_agent = self.agents[agent_i]
        zero_values = False
        # Train critic ------------------------
        curr_agent.critic_optimizer.zero_grad()
        
        if zero_values:
            all_trgt_acs = [torch.cat( # concat one-hot actions with params (that are zero'd along the indices of the non-chosen actions)
            (onehot_from_logits(pi(nobs)[:,:curr_agent.action_dim]),
             self.zero_params(pi(nobs),onehot_from_logits(pi(nobs)[:,:curr_agent.action_dim]))[:,curr_agent.action_dim:]),1)
                        for pi, nobs in zip(self.target_policies, next_obs)]    # onehot the action space but not param
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,next_obs)]  
            

        # Target critic values
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        #trgt_vf_in = torch.cat((*next_obs, *nacs), dim=1)

        # Actual critic values
        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        
        if self.D4PG:
            trgt_vf_distr = F.softmax(curr_agent.target_critic(trgt_vf_in),dim=1) # critic distribution
            trgt_vf_distr_proj = self.distr_projection(trgt_vf_distr,rews[agent_i],dones[agent_i],cum_rews[agent_i],
                                              gamma=self.gamma**self.REWARD_STEPS,device='cpu') 
            # distribution distance function
            prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
            vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
        else: # single critic value
            target_value = (1-self.beta)*(rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) * (1 - dones[agent_i].view(-1, 1))) + self.beta*(cum_rews[agent_i].view(-1,1))
            #vf_loss = MSELoss(actual_value, target_value)
            vf_loss = MSELoss(actual_value, target_value.detach())
            
        vf_loss.backward() 
        if parallel:
            average_gradients(curr_agent.critic)
        #torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 1)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()
        
        # Train actor -----------------------
        
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i]) # uses gumbel across the actions

            self.curr_pol_out = curr_pol_out.clone() # for inverting action space
            #gumbel = gumbel_softmax(torch.softmax(curr_pol_out[:,:curr_agent.action_dim].clone()),hard=True)
            #log_pol_out = curr_pol_out[:,:curr_agent.action_dim].clone()

            #gumbel = gumbel_softmax((curr_pol_out[:,:curr_agent.action_dim].clone()),hard=True)

            #log_pol_out = torch.log(curr_pol_out[:,:curr_agent.action_dim].clone())
            #gumbel = gumbel_softmax(log_pol_out,hard=True)
            #gumbel = onehot_from_logits(log_pol_out,eps=0.0)
            

            # concat one-hot actions with params zero'd along indices non-chosen actions
            if zero_values:
                curr_pol_vf_in = torch.cat((gumbel, 
                                      self.zero_params(curr_pol_out,gumbel)[:,curr_agent.action_dim:]),1)
            else:
                curr_pol_vf_in = curr_pol_out
                
            #print(curr_pol_vf_in)
        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else: # shariq does not gumbel this, we don't want to sample noise from other agents actions?
                a = pi(ob)
                g = onehot_from_logits(torch.log(a[:,:curr_agent.action_dim]),hard=True)
                c = torch.cat((g,self.zero_params(a,g)[:,curr_agent.action_dim:]),1)
                all_pol_acs.append(c) # 

        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        # invert gradient --------------------------------------
        self.params = vf_in.data
        self.param_dim = curr_agent.param_dim
        hook = vf_in.register_hook(self.inject)
        # ------------------------------------------------------
        if self.D4PG:
            critic_out = curr_agent.critic(vf_in)
            distr_q = curr_agent.critic.distr_to_q(critic_out)
            pol_loss = -distr_q.mean()
        else: # non-distributional
            pol_loss = -curr_agent.critic(vf_in).mean()
        #pol_loss += (curr_pol_out[:curr_agent.action_dim]**2).mean() * 1e-2 # regularize size of action
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        #torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_optimizer.step()
        hook.remove()

        # ------------------------------------
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

            
    def inject(self,grad):
        new_grad = grad.clone()
        new_grad = self.invert(new_grad,self.params,self.param_dim)
        #print("new",new_grad[0,-8:])
        return new_grad
    
    #zerod critic
    '''# takes input gradients and activation values for params and returns scaled gradients
    def invert(self,grad,params,num_params):
        for sample in range(grad.shape[0]): # batch size
            for index in range(num_params):
                if params[sample][-1 - index] != 0:
                # last 5 are the params
                    if grad[sample][-1 - index] < 0:
                        grad[sample][-1 - index] *= ((1.0-params[sample][-1 - index])/(1-(-1))) # scale
                    else:
                        grad[sample][-1 - index] *= ((params[sample][-1 - index]-(-1.0))/(1-(-1)))
                else:
                    grad[sample][-1-index] *= 0
        for sample in range(grad.shape[0]): # batch size
            # inverts gradients of discrete actions
            for index in range(3):
                if np.abs(grad[sample][-1-num_params -index]) > 10:
                    print(grad[sample][-1-num_params  -index])
                if params[sample][-1 - num_params - index] != 0:
                # last 5 are the params
                    if grad[sample][-1 - num_params - index] < 0:
                        grad[sample][-1 - num_params - index] *= ((1.0-self.curr_pol_out[sample][-1 - num_params -index])/(1-(-1))) # scale
                    else:
                        grad[sample][-1 - num_params - index] *= ((self.curr_pol_out[sample][-1 - num_params - index]-(-1.0))/(1-(-1)))
                else:
                    grad[sample][-1 - num_params - index] *= 0
            for index in range(3):
                if params[sample][-1-num_params-index] == 0:
                    grad[sample][-1-num_params-index] *= 0
        return grad'''
    
    # non-zerod critic
    # takes input gradients and activation values for params and returns scaled gradients
    def invert(self,grad,params,num_params):
        for sample in range(grad.shape[0]): # batch size
            for index in range(num_params):
            # last 5 are the params
                if grad[sample][-1 - index] < 0:
                    grad[sample][-1 - index] *= ((1.0-params[sample][-1 - index])/(1-(-1))) # scale
                else:
                    grad[sample][-1 - index] *= ((params[sample][-1 - index]-(-1.0))/(1-(-1)))
        for sample in range(grad.shape[0]): # batch size
            # inverts gradients of discrete actions
            for index in range(3):
            # last 5 are the params
                if grad[sample][-1 - num_params - index] < 0:
                    grad[sample][-1 - num_params - index] *= ((1.0-self.curr_pol_out[sample][-1 - num_params -index])/(1-(-1))) # scale
                else:
                    grad[sample][-1 - num_params - index] *= ((self.curr_pol_out[sample][-1 - num_params - index]-(-1.0))/(1-(-1)))

        return grad
 

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    #Needs to be tested
    def save_actor(self, filename):
        """
        Save trained parameters of all agent's actor network into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'actor_params': [a.get_actor_params() for a in self.agents]}
        torch.save(save_dict, filename)

    #Needs to be tested
    def save_critic(self, filename):
        """
        Save trained parameters of all agent's critic networks into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'critic_params': [a.get_critic_params() for a in self.agents]}
        torch.save(save_dict, filename)


    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, a_lr=0.01, c_lr=0.01, hidden_dim=64,discrete_action=True,
                      vmax = 10,vmin = -10, N_ATOMS = 51, REWARD_STEPS = 5, DELTA_Z = 20.0/50,D4PG=False,beta=0):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        
        alg_types = [ agent_alg for
                     atype in range(env.num_TA)]
        for acsp, obsp, algtype in zip([env.action_list for i in range(env.num_TA)], env.team_obs, alg_types):
            
            # changed acsp to be action_list for each agent 
                # giving dimension num_TA x action_list so they may zip properly    

            num_in_pol = obsp.shape[0]
        
            num_out_pol =  len(env.action_list)
            
            
    
            # if cont
            if not discrete_action:
                num_out_pol = len(env.action_list) + len(env.action_params[0])
                
                
            # obs space and action space are concatenated before sending to
            # critic network
            num_in_critic = (num_in_pol + num_out_pol) *env.num_TA
            
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
            
        ## change for continuous
        init_dict = {'gamma': gamma, 'tau': tau, 'a_lr': a_lr,
                     'c_lr':c_lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'vmax': vmax,
                     'vmin': vmin,
                     'N_ATOMS': N_ATOMS,
                     'REWARD_STEPS': REWARD_STEPS,
                     'DELTA_Z': DELTA_Z,
                     'D4PG': D4PG,
                     'beta': beta}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
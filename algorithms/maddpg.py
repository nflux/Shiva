import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork_Actor,MLPNetwork_Critic
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax,hard_update,zero_params,distr_projection,processor
from utils.agents import DDPGAgent
import numpy as np
import random
from torch.autograd import Variable
import os
import time
import torch.multiprocessing as mp
import dill
import copy
import pandas as pd
MSELoss = torch.nn.MSELoss()
CELoss = torch.nn.CrossEntropyLoss()

def init_from_env(config, env):
    maddpg = MADDPG.init_from_env(env, config)

    if config.multi_gpu:
        maddpg.torch_device = torch.device("cuda:0")
    
    if config.to_gpu:
        maddpg.device = 'cuda'
    maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
    
    if config.first_save: # Generate list of ensemble networks
        file_path = config.ensemble_path
        maddpg.first_save(file_path,num_copies = config.k_ensembles)
        [maddpg.save_agent(config.load_path,0,i,load_same_agent = False,torch_device=maddpg.torch_device) for i in range(config.num_left)] 
        if config.preload_model:
            maddpg.load_team(side='team',models_path=config.preload_path,nagents=config.num_left)
            maddpg.first_save(file_path,num_copies = config.k_ensembles) 

        config.first_save = False
    
    return maddpg

def parallel_step(results,a_i,ran,obs,explore,output,pi_pickle,action_dim=3,param_dim=5,device='cpu',exploration_noise=0.000001):

    pi = dill.loads(pi_pickle)

    if explore:
        if not ran: # not random
            action = pi(obs)
            a = gumbel_softmax(action[0,:action_dim].view(1,action_dim),hard=True, device=device)
            p = torch.clamp((action[0,action_dim:].view(1,param_dim) + Variable(processor(torch.Tensor(exploration_noise),device=device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
            action = torch.cat((a,p),1) 
        else: # random
            action = torch.cat((onehot_from_logits(torch.empty((1,action_dim),device=device,requires_grad=False).uniform_(-1,1)),
                        torch.empty((1,param_dim),device=device,requires_grad=False).uniform_(-1,1) ),1)
    else:
        action = pi(obs)
        a = onehot_from_logits(action[0,:action_dim].view(1,action_dim))
        #p = torch.clamp(action[0,self.action_dim:].view(1,self.param_dim),min=-1.0,max=1.0) # get noisey params (OU)
        p = torch.clamp((action[0,action_dim:].view(1,param_dim) + Variable(processor(torch.Tensor(exploration_noise),device=device),requires_grad=False)),min=-1.0,max=1.0) # get noisey params (OU)
        action = torch.cat((a,p),1) 
    results[a_i] = action
 
class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, team_agent_init_params, opp_agent_init_params, team_net_params, team_alg_types='MADDPG', opp_alg_types='MADDPG',device='cpu',
                 gamma=0.95, batch_size=0,tau=0.01, a_lr=0.01, c_lr=0.01, hidden_dim=64,
                 discrete_action=True,vmax = 10,vmin = -10, N_ATOMS = 51, n_steps = 5,
                 DELTA_Z = 20.0/50,D4PG=False,beta = 0,TD3=False,TD3_noise = 0.2,TD3_delay_steps=2,
                 I2A = False,EM_lr = 0.001,obs_weight=10.0,rew_weight=1.0,ws_weight=1.0,rollout_steps = 5,
                 LSTM_hidden=64, imagination_policy_branch = False,
                 critic_mod_both=False, critic_mod_act=False, critic_mod_obs=False,
                 LSTM=False, LSTM_policy=False,seq_length = 20, hidden_dim_lstm=256, lstm_burn_in=40,overlap=20,only_policy=False,
                 multi_gpu=True,data_parallel=False,reduced_obs_dim=16,preprocess=False,zero_critic=False,cent_critic=True): 

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
        self.world_status_dim = 8 # number of possible world statuses
        self.nagents_team = len(team_alg_types)
        self.nagents_opp = len(opp_alg_types)
        self.critic_loss_logger = pd.DataFrame()
        self.policy_loss_logger = pd.DataFrame()
        self.team_alg_types = team_alg_types
        self.opp_alg_types = opp_alg_types
        self.num_in_EM = team_net_params[0]['num_in_EM']
        self.num_out_EM = team_net_params[0]['num_out_EM']
        self.batch_size = batch_size
        self.only_policy = only_policy
        self.multi_gpu = multi_gpu
        self.data_parallel = data_parallel
        self.num_out_pol = team_net_params[0]['num_out_pol']
        self.team_agent_init_params = team_agent_init_params
        self.opp_agent_init_params = opp_agent_init_params
        self.team_net_params = team_net_params
        self.device = device
        self.torch_device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.EM_dev = 'cpu'  # device for EM
        self.prime_dev = 'cpu'  # device for pol prime
        self.imagination_pol_dev = 'cpu'# device for imagination branch policy
        self.cent_critic = cent_critic
        self.preprocess = preprocess
        self.zero_critic = zero_critic
        self.LSTM = LSTM
        self.LSTM_policy = LSTM_policy
        self.lstm_burn_in = lstm_burn_in
        self.hidden_dim_lstm = hidden_dim_lstm
        self.seq_length = seq_length
        self.overlap = overlap
        self.niter = 0
        self.n_steps = n_steps
        self.beta = beta
        self.N_ATOMS = N_ATOMS
        self.Vmax = vmax
        self.Vmin = vmin
        self.DELTA_Z = DELTA_Z
        self.D4PG = D4PG
        self.TD3 = TD3
        self.TD3_noise = TD3_noise
        self.TD3_delay_steps = TD3_delay_steps
        if not TD3:
            self.TD3_delay_steps = 1
        self.EM_lr = EM_lr
        self.I2A = I2A
        self.obs_weight = obs_weight
        self.rew_weight = rew_weight
        self.ws_weight = ws_weight
        self.ws_onehot = processor(torch.FloatTensor(self.batch_size,self.world_status_dim),device=self.device,torch_device=self.torch_device) 
        self.team_count = [0 for i in range(self.nagents_team)]
        self.opp_count = [0 for i in range(self.nagents_opp)]
        self.team_agents = [DDPGAgent(discrete_action=discrete_action, maddpg=self,
                                 hidden_dim=hidden_dim,a_lr=a_lr, c_lr=c_lr,
                                 n_atoms = N_ATOMS, vmax = vmax, vmin = vmin,
                                 delta = DELTA_Z,D4PG=D4PG,
                                 TD3=TD3,
                                 I2A = I2A,EM_lr=EM_lr,
                                 world_status_dim=self.world_status_dim,
                                      rollout_steps = rollout_steps,LSTM_hidden=LSTM_hidden,
                                      device=device,
                                      imagination_policy_branch=imagination_policy_branch,
                                      critic_mod_both = critic_mod_both, critic_mod_act=critic_mod_act, critic_mod_obs=critic_mod_obs,
                                      LSTM=LSTM, LSTM_policy=LSTM_policy,seq_length=seq_length, hidden_dim_lstm=hidden_dim_lstm,reduced_obs_dim=reduced_obs_dim,
                                      
                                 **params)
                       for params in team_agent_init_params]
        
        self.opp_agents = [DDPGAgent(discrete_action=discrete_action, maddpg=self,
                                 hidden_dim=hidden_dim,a_lr=a_lr, c_lr=c_lr,
                                 n_atoms = N_ATOMS, vmax = vmax, vmin = vmin,
                                 delta = DELTA_Z,D4PG=D4PG,
                                 TD3=TD3,
                                 I2A = I2A,EM_lr=EM_lr,
                                 world_status_dim=self.world_status_dim,
                                     rollout_steps = rollout_steps,LSTM_hidden=LSTM_hidden,device=device,
                                     imagination_policy_branch=imagination_policy_branch,
                                     critic_mod_both = critic_mod_both, critic_mod_act=critic_mod_act, critic_mod_obs=critic_mod_obs,
                                     LSTM=LSTM, LSTM_policy=LSTM_policy, seq_length=seq_length, hidden_dim_lstm=hidden_dim_lstm,reduced_obs_dim=reduced_obs_dim,
                                 **params)
                       for params in opp_agent_init_params]


    @property
    def team_policies(self):
        return [a.policy for a in self.team_agents]
    
    @property
    def opp_policies(self):
        return [a.policy for a in self.opp_agents]

    @property
    def team_target_policies(self):
        return [a.target_policy for a in self.team_agents]

    @property
    def opp_target_policies(self):
        return [a.target_policy for a in self.opp_agents]
    
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
        for a in self.team_agents:
            a.scale_noise(scale)

        for a in self.opp_agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.team_agents:
            a.reset_noise()
        
        for a in self.opp_agents:
            a.reset_noise()

    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def zero_hidden(self, batch_size,actual=False,target=False,torch_device = torch.device('cuda:0')):
        if actual:
            self.team_agents[0].critic.init_hidden(batch_size,torch_device=torch_device)
        if target:
            self.team_agents[0].target_critic.init_hidden(batch_size,torch_device=torch_device)

        # for ta, oa in zip(self.team_agents, self.opp_agents):
        #     ta.critic.init_hidden(batch_size)
        #     oa.critic.init_hidden(batch_size)

    def zero_hidden_policy(self, batch_size,torch_device=torch.device('cuda:0')):
        [a.policy.init_hidden(batch_size,torch_device=torch_device) for a in self.team_agents]
        [a.policy.init_hidden(batch_size,torch_device=torch_device) for a in self.opp_agents]
        [a.target_policy.init_hidden(batch_size,torch_device=torch_device) for a in self.team_agents]
        [a.target_policy.init_hidden(batch_size,torch_device=torch_device) for a in self.opp_agents]
        # for ta, oa in zip(self.team_agents, self.opp_agents):
        #     ta.critic.init_hidden(batch_size)
        #     oa.critic.init_hidden(batch_size)
    
    def cast_hidden(self, tensor,torch_device=torch.device('cuda:0')):
        if self.device == 'cuda':
            h1 = (Variable(tensor[None, :, :self.hidden_dim_lstm]).to(torch_device),
                                            Variable(tensor[None, :, self.hidden_dim_lstm:self.hidden_dim_lstm*2]).to(torch_device))
            h2 = (Variable(tensor[None, :, self.hidden_dim_lstm*2:self.hidden_dim_lstm*3]).to(torch_device),
                                            Variable(tensor[None, :, self.hidden_dim_lstm*3:self.hidden_dim_lstm*4]).to(torch_device))
        else:
            h1 = (Variable(tensor[None, :, :self.hidden_dim_lstm]),
                                            Variable(tensor[None, :, self.hidden_dim_lstm:self.hidden_dim_lstm*2]))
            h2 = (Variable(tensor[None, :, self.hidden_dim_lstm*2:self.hidden_dim_lstm*3]),
                                            Variable(tensor[None, :, self.hidden_dim_lstm*3:self.hidden_dim_lstm*4]))
        
        return h1, h2
    
    def set_hidden(self, h1, h2, actual=False,target=False,torch_device=torch.device('cuda:0')):
        if actual:
            self.team_agents[0].critic.set_hidden(h1, h2)
        if target:
            self.team_agents[0].target_critic.set_hidden(h1,h2)

    def step(self, team_observations, opp_observations,team_e_greedy,opp_e_greedy,parallel, explore=False,LSTM_policy=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        if self.I2A:
            if LSTM_policy:
                team_acs, rec_state = self.team_agents[0].policy(team_observations)
                opp_acs,opp_rec_state = self.opp_agents[0].policy(opp_observations)
            else:
                team_acs = self.team_agents[0].policy(team_observations)
                opp_acs = self.opp_agents[0].policy(opp_observations)

            return [a.step(obs,ran, acs,explore=explore) for a,ran, obs,acs in zip(self.team_agents, team_e_greedy,team_observations,team_acs)], \
                    [a.step(obs,ran, acs,explore=explore) for a,ran, obs,acs in zip(self.opp_agents,opp_e_greedy, opp_observations,opp_acs)]


        else:                
            return [a.step(obs,ran, explore=explore) for a,ran, obs in zip(self.team_agents, team_e_greedy,team_observations)], \
                    [a.step(obs,ran, explore=explore) for a,ran, obs in zip(self.opp_agents,opp_e_greedy, opp_observations)]

    def get_recurrent_states(self, exps, obs_dim, acs_dim, nagents, hidden_dim_lstm,torch_device):
        ep_length = len(exps)
        self.zero_hidden(1,actual=True,target=True,torch_device=torch_device)
        

        for e in range(0, ep_length-self.seq_length, self.overlap):
            # Assumes M vs M
            if (e + self.overlap + self.seq_length) <= ep_length:
                self.zero_hidden(1,actual=True,target=True,torch_device=torch_device)

                obs = exps[e:e+self.overlap, :, :obs_dim]
                acs = exps[e:e+self.overlap, :, obs_dim:obs_dim+acs_dim]

                obs = obs.reshape(self.overlap, 1, nagents*obs_dim)
                acs = acs.reshape(self.overlap, 1, nagents*acs_dim)

                comb = torch.cat((obs, acs), dim=2).to(torch_device) # critic device
                _,_,hs1,hs2 = self.team_agents[0].critic(comb.float())
                # self.set_hidden(hs1, hs2)
                self.repackage_hidden(hs1)
                self.repackage_hidden(hs2)
                temp_all_hs =  torch.cat((hs1[0].squeeze(), hs1[1].squeeze(), hs2[0].squeeze(), hs2[1].squeeze()), dim=0)
                self.zero_hidden(1,actual=True,target=True,torch_device=torch_device)

                exps[e+self.overlap, :, -hidden_dim_lstm*4:] = temp_all_hs
            else:
                self.zero_hidden(1,actual=True,target=True,torch_device=torch_device)
                obs = exps[:ep_length-self.seq_length, :, :obs_dim]
                acs = exps[:ep_length-self.seq_length, :, obs_dim:obs_dim+acs_dim]

                obs = obs.reshape(ep_length-self.seq_length, 1, nagents*obs_dim)
                acs = acs.reshape(ep_length-self.seq_length, 1, nagents*acs_dim)

                comb = torch.cat((obs, acs), dim=2).to(torch_device)

                _,_,hs1,hs2 = self.team_agents[0].critic(comb.float())
                self.repackage_hidden(hs1)
                self.repackage_hidden(hs2)
                # self.set_hidden(hs1, hs2)

                temp_all_hs =  torch.cat((hs1[0].squeeze(), hs1[1].squeeze(), hs2[0].squeeze(), hs2[1].squeeze()), dim=0)
                exps[ep_length-self.seq_length, :, -hidden_dim_lstm*4:].copy_(temp_all_hs)
                temp_all_hs.detach_()
                self.zero_hidden(1,actual=True,target=True,torch_device=torch_device)
                del temp_all_hs
                torch.cuda.empty_cache()





    def discrete_param_indices(self,discrete):
        if discrete == 0:
            return [0,1]
        elif discrete == 1:
            return [2]
        if discrete == 2:
            return [3,4]
                     
                     
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


    def update(self, sample, agent_i, side='team', parallel=False, logger=None):
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
        obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = sample


        self.curr_agent_index = agent_i
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies

        
        # Train critic ------------------------
        curr_agent.critic_optimizer.zero_grad()


        if self.TD3:
            noise = processor(torch.randn_like(acs[0]),device=self.device) * self.TD3_noise
            all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in [(pi(nobs) + noise) for pi, nobs in zip(target_policies,next_obs)]]
        else:
            all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in [pi(nobs) for pi, nobs in zip(target_policies,next_obs)]]


        # Target critic values
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        if self.TD3: # TODO* For D4PG case, need mask with indices of the distributions whos distr_to_q(trgtQ1) < distr_to_q(trgtQ2)
                     # and build the combination of distr choosing the minimums
            trgt_Q1,trgt_Q2 = curr_agent.target_critic(trgt_vf_in)
            if self.D4PG:
                arg = torch.argmin(torch.stack((curr_agent.target_critic.distr_to_q(trgt_Q1).mean(),
                                 curr_agent.target_critic.distr_to_q(trgt_Q2).mean()),dim=0))

                if not arg: 
                    trgt_Q = trgt_Q1
                else:
                    trgt_Q = trgt_Q2
            else:
                trgt_Q = torch.min(trgt_Q1,trgt_Q2)
        else:
            trgt_Q = curr_agent.target_critic(trgt_vf_in)
        # Actual critic values
        vf_in = torch.cat((*obs, *acs), dim=1)
        if self.TD3:
            actual_value_1, actual_value_2 = curr_agent.critic(vf_in)
        else:
            actual_value = curr_agent.critic(vf_in)
        if self.D4PG:
                # Q1
                trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                  gamma=self.gamma**self.n_steps,device=self.device)
                if self.TD3:
                    prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                    prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                    # distribution distance function
                    vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                else:
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                      gamma=self.gamma**self.n_steps,device=self.device) 
                    # distribution distance function
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
        else: # single critic value
            target_value = (1-self.beta)*(n_step_rews[agent_i].view(-1, 1) + (self.gamma**self.n_steps) *
                        trgt_Q * (1 - dones[agent_i].view(-1, 1))) + self.beta*(MC_rews[agent_i].view(-1,1))
            target_value.detach()
            if self.TD3: # handle double critic
                vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
            else:
                vf_loss = F.mse_loss(actual_value, target_value)

                
            
        vf_loss.backward() 
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()
        
        # Train actor -----------------------
        if count % self.TD3_delay_steps == 0:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = torch.cat((gumbel_softmax(curr_pol_out[:,:curr_agent.action_dim], hard=True, device=self.device),curr_pol_out[:,curr_agent.action_dim:]),dim=1)
            team_pol_acs = []
            for i, pi, ob in zip(range(nagents), policies, obs):
                if i == agent_i:
                    team_pol_acs.append(curr_pol_vf_in)
                else: # shariq does not gumbel this, we don't want to sample noise from other agents actions?
                    a = pi(ob)
                    team_pol_acs.append(torch.cat((onehot_from_logits(a[:,:curr_agent.action_dim]),a[:,curr_agent.action_dim:]),dim=1))
            vf_in = torch.cat((*obs, *team_pol_acs), dim=1)

            # ------------------------------------------------------
            if self.D4PG:
                critic_out = curr_agent.critic.Q1(vf_in)
                distr_q = curr_agent.critic.distr_to_q(critic_out)
                pol_loss = -distr_q.mean()
            else: # non-distributional
                pol_loss = -curr_agent.critic.Q1(vf_in).mean()              
      

            param_reg = torch.clamp((curr_pol_out[:,curr_agent.action_dim:]**2)-torch.ones_like(curr_pol_out[:,curr_agent.action_dim:]),min=0.0).sum(dim=1).mean()
            entropy_reg = (-torch.log_softmax(curr_pol_out,dim=1)[:,:curr_agent.action_dim].sum(dim=1).mean() * 1e-3)/5.0 # regularize using log probabilities
            #pol_loss += ((curr_pol_out**2)[:,:curr_agent.action_dim].mean() * 1e-3)/3.0 #Shariq-style regularizer on size of linear outputs
            pol_loss += param_reg
            pol_loss += entropy_reg
            #pol_loss.backward(retain_graph=True)
            pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_optimizer.step()
            if self.niter % 100 == 0:
                print("Team (%s) Agent (%i) Actor loss:" % (side,agent_i),pol_loss)



        # I2A --------------------------------------
        if self.I2A:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            curr_pol_out = curr_agent.policy(obs[agent_i])
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = curr_agent.policy_prime(obs[agent_i]) # uses gumbel across the actions
            pol_prime_out_actions = pol_prime_out[:,:curr_agent.action_dim].float()
            pol_prime_out_params = pol_prime_out[:,curr_agent.action_dim:]
            pol_out_actions = curr_pol_out[:,:curr_agent.action_dim].float()
            pol_out_params = curr_pol_out[:,curr_agent.action_dim:]
            target_classes = torch.argmax(pol_out_actions,dim=1) # categorical integer for predicted class
            #MSE =np.sum([F.mse_loss(prime[self.discrete_param_indices(target_class)],current[self.discrete_param_indices(target_class)]) for prime,current,target_class in zip(pol_prime_out_params,pol_out_params, target_classes)])/(1.0*self.batch_size)
            MSE = F.mse_loss(pol_prime_out_params,pol_out_params)
            #pol_prime_loss = MSE + 
            pol_prime_loss = MSE + F.mse_loss(pol_prime_out_actions,pol_out_actions)
            pol_prime_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy_prime)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()

            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            self.ws_onehot.zero_() # reset OH tensor
            self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            if self.decent_EM:
                EM_in = torch.cat((*[obs[agent_i]], *[acs[agent_i]]),dim=1)
            else:
                EM_in = torch.cat((*obs,*acs),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = next_obs[agent_i] - obs[agent_i]
            actual_rews = rews[agent_i].view(-1,1)
            actual_ws = self.ws_onehot
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew + self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()

            #---------------------------------------------------------------------------------
        if side == 'team':
            self.team_count[agent_i] += 1
        else:
            self.opp_count[agent_i] += 1
        # ------------------------------------
        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)

            
        if self.niter % 100 == 0:
            print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)
            if self.I2A:
                print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)

                
    def update_centralized_critic(self, team_sample=[], opp_sample =[], agent_i = 0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,forward_pass=True,load_same_agent=False,critic=True,policy=True,session_path=""):
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
        if self.niter % 100:
            if critic and policy:
                self.critic_loss_logger.to_csv(session_path + 'loss.csv')
            elif critic:
                self.critic_loss_logger.to_csv(session_path + 'critic_loss.csv')
            elif policy:
                self.policy_loss_logger.to_csv(session_path + 'actor_loss.csv')

        start = time.time()
        
        #start = time.time()
        # rews = 1-step, cum-rews = n-step
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            opp_target_policies = self.opp_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = team_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            opp_target_policies = self.team_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = opp_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = team_sample

            
        
        self.curr_agent_index = agent_i
        if self.preprocess:
            reducer = curr_agent.reducer
            red_obs = [reducer.reduce(o) for o in obs]
            red_next_obs = [reducer.reduce(no) for no in next_obs]
            red_opp_obs = [reducer.reduce(oo) for oo in opp_obs]
            red_opp_next_obs = [reducer.reduce(ono) for ono in opp_next_obs]
        # Train critic ------------------------
        if critic:
            curr_agent.critic_optimizer.zero_grad()
            if load_same_agent:
                curr_agent = self.team_agents[0]
            #print("time critic")
            #start = time.time()
            #with torch.no_grad():
            if self.TD3:
                noise = processor(torch.randn_like(acs[0]),device=self.device,torch_device=self.torch_device) * self.TD3_noise
                if self.I2A:
                    if self.preprocess:
                        team_pi_acs = [a + noise for a in target_policies[0](red_next_obs)] # get actions for all agents, add noise to each
                        opp_pi_acs = [a + noise for a in opp_target_policies[0](red_opp_next_obs)] # get actions for all agents, add noise to each
                    else:
                        team_pi_acs = [a + noise for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                        opp_pi_acs = [a + noise for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each

                else:
                    if self.preprocess:
                        team_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(target_policies,red_next_obs)]
                        opp_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(opp_target_policies,red_opp_next_obs)]
                    else:
                        team_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(target_policies,next_obs)]
                        opp_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(opp_target_policies,opp_next_obs)]


                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in team_pi_acs]

                opp_all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in opp_pi_acs]
            else:
                if self.I2A:
                    if self.preprocess:
                        team_pi_acs = [a for a in target_policies[0](red_next_obs)] # get actions for all agents, add noise to each
                        opp_pi_acs = [a for a in opp_target_policies[0](red_opp_next_obs)] # get actions for all agents, add noise to each
                    else:
                        team_pi_acs = [a for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                        opp_pi_acs = [a for a in opp_target_policies[0](next_obs)] # get actions for all agents, add noise to each
                else:
                    if self.preprocess:
                        team_pi_acs  = [(pi(nobs)) for pi, nobs in zip(target_policies,red_next_obs)]
                        opp_pi_acs  = [(pi(nobs)) for pi, nobs in zip(opp_target_policies,red_opp_next_obs)]
                    else:
                        team_pi_acs  = [(pi(nobs)) for pi, nobs in zip(target_policies,next_obs)]
                        opp_pi_acs  = [(pi(nobs)) for pi, nobs in zip(opp_target_policies,opp_next_obs)]

                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in team_pi_acs]
                opp_all_trgt_acs =[torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in opp_pi_acs]
                

            if self.zero_critic:
                all_trgt_acs = [zero_params(a) for a in all_trgt_acs]
                opp_all_trgt_acs = [zero_params(a) for a in opp_all_trgt_acs]
                
                
            if self.preprocess:
                mod_next_obs = torch.cat((*red_next_obs,*red_opp_next_obs),dim=1)
            else:
                mod_next_obs = torch.cat((*next_obs,*opp_next_obs),dim=1)
            
            mod_all_trgt_acs = torch.cat((*all_trgt_acs,*opp_all_trgt_acs),dim=1)

            # Target critic values
            trgt_vf_in = torch.cat((mod_next_obs, mod_all_trgt_acs), dim=1)
            if self.TD3: # TODO* For D4PG case, need mask with indices of the distributions whos distr_to_q(trgtQ1) < distr_to_q(trgtQ2)
                        # and build the combination of distr choosing the minimums
                trgt_Q1,trgt_Q2 = curr_agent.target_critic(trgt_vf_in)
                if self.D4PG:
                    arg = torch.argmin(torch.stack((curr_agent.target_critic.distr_to_q(trgt_Q1).mean(),
                                    curr_agent.target_critic.distr_to_q(trgt_Q2).mean()),dim=0))

                    if not arg: 
                        trgt_Q = trgt_Q1
                    else:
                        trgt_Q = trgt_Q2
                else:
                    trgt_Q = torch.min(trgt_Q1,trgt_Q2)
            else:
                trgt_Q = curr_agent.target_critic(trgt_vf_in)
            
            if self.preprocess:
                mod_obs = torch.cat((*red_obs,*red_opp_obs),dim=1)
            else:
                mod_obs = torch.cat((*obs,*opp_obs),dim=1)
            if self.zero_critic:
                mod_acs = torch.cat((*[zero_params(a) for a in acs],*[zero_params(a) for a in opp_acs]),dim=1)
            else:
                mod_acs = torch.cat((*acs,*opp_acs),dim=1)

            # Actual critic values
            vf_in = torch.cat((mod_obs, mod_acs), dim=1)

            if self.TD3:
                actual_value_1, actual_value_2 = curr_agent.critic(vf_in)
            else:
                actual_value = curr_agent.critic(vf_in)
            
            if self.D4PG:
                    # Q1
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution

                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device)

                    if self.TD3:
                        prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                        prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                        # distribution distance function
                        vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                    else:
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                        trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                        gamma=self.gamma**self.n_steps,device=self.device) 
                        # distribution distance function
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
            else: # single critic value
                target_value = (1-self.beta)*(torch.cat([n.view(-1,1) for n in n_step_rews],dim=1).float().mean(dim=1).view(-1, 1) + (self.gamma**self.n_steps) *
                            trgt_Q * (1 - dones[agent_i].view(-1, 1))) + self.beta*(torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1)).view(-1,1)
                target_value.detach()

                if self.TD3: # handle double critic

                    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                    prio = np.round(prio.cpu().numpy(),decimals=3)
                    prio = torch.tensor(prio,requires_grad=False)
                    vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
                else:
                    vf_loss = F.mse_loss(actual_value, target_value)

                            

            #vf_loss.backward()
            vf_loss.backward() 
            
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
            curr_agent.critic_optimizer.step()

            # Train preprocessor ---------------------------------
            if self.preprocess:
                if self.niter % 1 == 0:
                    curr_agent.reducer_optimizer.zero_grad()
                    obs_stacked = torch.cat(obs,dim=0)
                    rec_obs_stacked = reducer(obs_stacked)

                    opp_obs_stacked = torch.cat(opp_obs,dim=0)
                    rec_opp_obs_stacked = reducer(opp_obs_stacked)
                    preprocess_loss = F.mse_loss(rec_obs_stacked,obs_stacked)
                    preprocess_loss += F.mse_loss(rec_opp_obs_stacked,opp_obs_stacked)
                    preprocess_loss.backward()
                    curr_agent.reducer_optimizer.step()


            
            
        if policy:
            curr_agent.policy_optimizer.zero_grad()
    

            # Train actor -----------------------
                    
            #print("time actor")
            team_pol_acs = []
    
            if self.I2A:
                if self.preprocess:
                    curr_pol_out = curr_agent.policy(red_obs)
                else:
                    curr_pol_out = curr_agent.policy(obs)

            else:
                if self.preprocess:
                    curr_pol_out = [curr_agent.policy(red_obs[ag]) for ag in range(nagents)]
                else:
                    curr_pol_out = [curr_agent.policy(obs[ag]) for ag in range(nagents)]

                
            team_pol_acs = [torch.cat((gumbel_softmax(c[:,:curr_agent.action_dim], hard=True, device=self.torch_device),c[:,curr_agent.action_dim:]),dim=1) for c in curr_pol_out]
            curr_pol_out_stacked = torch.cat(curr_pol_out,dim=0)
            if self.zero_critic:
                curr_pol_out_stacked = zero_params(curr_pol_out_stacked)

            if self.preprocess:
                obs_vf_in = torch.cat((*red_obs,*red_opp_obs),dim=1)
            else:
                obs_vf_in = torch.cat((*obs,*opp_obs),dim=1)

            acs_vf_in = torch.cat((*team_pol_acs,*opp_acs),dim=1)
            
            if self.zero_critic:
                acs_vf_in = torch.cat((*[zero_params(a) for a in team_pol_acs],*[zero_params(a) for a in opp_acs]),dim=1)
            mod_vf_in = torch.cat((obs_vf_in, acs_vf_in), dim=1)

            # ------------------------------------------------------
            if self.D4PG:
                if self.data_parallel:
                    critic_out = curr_agent.critic.module.Q1(mod_vf_in)
                else:
                    critic_out = curr_agent.critic.Q1(mod_vf_in)
                distr_q = curr_agent.critic.distr_to_q(critic_out)
                pol_loss = -distr_q.mean()
            else: # non-distributional
                if self.data_parallel:
                    pol_loss = -curr_agent.critic.module.Q1(mod_vf_in).mean() 
                else:
                    pol_loss = -curr_agent.critic.Q1(mod_vf_in).mean() 
    
            if self.D4PG:
                reg_param = 5.0
            else:
                reg_param = 5.0
            
            param_reg = torch.clamp((curr_pol_out_stacked[:,curr_agent.action_dim:]**2)-torch.ones_like(curr_pol_out_stacked[:,curr_agent.action_dim:]),min=0.0).sum(dim=1).mean()
            entropy_reg = (-torch.log_softmax(curr_pol_out_stacked[:,:curr_agent.action_dim],dim=1).sum(dim=1).mean() * 1e-3)/reg_param # regularize using log probabilities
            pol_loss += param_reg
            pol_loss += entropy_reg
            if self.I2A:
                pol_loss.backward(retain_graph = True)
            else:
                pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_optimizer.step()

                #print(time.time() - start,"update time")
        # I2A --------------------------------------
        if self.I2A and policy:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            if self.preprocess:
                pol_prime_out = [curr_agent.policy_prime(red_obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            else:
                pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions

            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            #labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            #self.ws_onehot.zero_() # reset OH tensor
            #self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            if self.preprocess:
                agents_obs = torch.cat(red_obs,dim=0) # stack entire batch on top of eachother for each agent
                agents_nobs = torch.cat(red_next_obs,dim=0)
            else:
                agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
                agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch
            
            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            #actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew# + self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        if policy:
            if self.niter % 100 == 0:
                print("Team (%s) Agent (%i) Actor loss:" % (side,agent_i),pol_loss)


                if self.I2A:
                    print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                    print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)

                    
                    
    

                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4),
                                                                        'prime_loss': np.round(pol_prime_loss.item(),4),
                                                                        'em_loss': np.round(EM_loss.item(),4)},
                                                                        ignore_index=True)
                else:
                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4)},                    
                                                                        ignore_index=True)
            #-------
            #---------------------------------------------------------------------------------
        if side == 'team':
            self.team_count[agent_i] += 1
        else:
            self.opp_count[agent_i] += 1
        # ------------------------------------
        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)
        #print(time.time() - start,"up")
        if critic:
            if self.niter % 100 == 0:
                if self.preprocess:
                    self.critic_loss_logger = self.critic_loss_logger.append({                          'iteration':self.niter,
                                                                            'preprocess':np.round(preprocess_loss.item(),4),
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                    print("Team (%s) Agent(%i) preprocessor loss" % (side, agent_i),preprocess_loss)

                else:
                    self.critic_loss_logger = self.critic_loss_logger.append({                          'iteration':self.niter,
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)

        self.niter +=1


        # return priorities
        if critic:
            if self.TD3:
                if self.D4PG:
                    return ((prob_dist_1.float().sum(dim=1) + prob_dist_2.float().sum(dim=1))/2.0).cpu()
                else:
                    return prio
                    
            else:
                return prob_dist.sum(dim=1).cpu()
        else:
            return None

    def update_LSTM_old(self, team_sample=[], opp_sample=[], agent_i=0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,
                                        forward_pass=True,load_same_agent=False,critic=False,policy=True,session_path="",lstm_burn_in=40):
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
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            opp_target_policies = self.opp_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            _, _, rews, dones, MC_rews,n_step_rews,ws,rec_states,sorted_feats = team_sample
            _, _, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_,_ = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            opp_target_policies = self.team_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, dones,MC_rews,n_step_rews,ws,rec_states,_ = opp_sample
            opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_,_ = team_sample
        self.curr_agent_index = agent_i
        if load_same_agent:
            curr_agent = self.team_agents[0]
        # Train critic ------------------------
        if critic:

            obs = sorted_feats[0][0] #use features sorted by prox and stacked per agent along batch
            opp_obs = sorted_feats[0][1]
            acs = sorted_feats[0][2]
            opp_acs = sorted_feats[0][3] 
            #print(acs)
            #print(opp_acs)
            curr_agent.critic_optimizer.zero_grad()
            self.zero_hidden(self.batch_size,actual=True,target=True,torch_device=self.torch_device)
            self.zero_hidden_policy(self.batch_size,torch_device=self.torch_device)
            #h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device)
            #self.set_hidden(h1, h2,actual=True,target=True,torch_device=self.torch_device)
            burnin_slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            burnin_slice_acs = list(map(lambda x: x[:lstm_burn_in], acs))
            burnin_slice_obs_opp = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            burnin_slice_acs_opp = list(map(lambda x: x[:lstm_burn_in], opp_acs))
            if self.zero_critic:
                burnin_slice_acs = [zero_params(a) for a in burnin_slice_acs]
                burnin_slice_acs_opp = [zero_params(a) for a in burnin_slice_acs_opp]
            
            # Run burn-in on target policy
            burnin_slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            [pi(nobs) for pi,nobs in zip(target_policies,burnin_slice_obs)]# burn in target
            # Run burn-in on opponent target policy
            burnin_opp_slice_obs = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            [pi(nobs) for pi,nobs in zip(opp_target_policies,burnin_slice_obs_opp)]# burn in opp target

            burnin_mod_obs = torch.cat((*burnin_slice_obs, *burnin_slice_obs_opp), dim=2)
            burnin_mod_acs = torch.cat((*burnin_slice_acs, *burnin_slice_acs_opp), dim=2)
            burn_in_tensor = torch.cat((burnin_mod_obs,burnin_mod_acs), dim=2)
            # #Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burn_in_tensor)
            _,_,h1_target,h2_target = curr_agent.target_critic(burn_in_tensor)
            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs))
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))
            n_step_rews = list(map(lambda x: x[lstm_burn_in:], n_step_rews))
            MC_rews = list(map(lambda x: x[lstm_burn_in:], MC_rews))
            dones = list(map(lambda x: x[lstm_burn_in:], dones))
            next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), obs))
            opp_next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), opp_obs))
            start = time.time()
            if self.TD3:
                noise = processor(torch.randn_like(acs[0][lstm_burn_in-1:]),device=self.device,torch_device=self.torch_device) * self.TD3_noise
                if self.I2A:
                    team_pi_acs = [a + noise for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a + noise for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(opp_target_policies,opp_next_obs)]
                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=True),out[:,:,curr_agent.action_dim:]),dim=2) for out in team_pi_acs]
                opp_all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=True),out[:,:,curr_agent.action_dim:]),dim=2) for out in opp_pi_acs]
            else:
                if self.I2A:
                    team_pi_acs = [a for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs)) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs)) for pi, nobs in zip(opp_target_policies,opp_next_obs)]
                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in team_pi_acs]
                opp_all_trgt_acs =[torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in opp_pi_acs]
            if self.zero_critic:
                all_trgt_acs = [zero_params(a) for a in all_trgt_acs]
                opp_all_trgt_acs = [zero_params(a) for a in opp_all_trgt_acs]
                
            mod_next_obs = torch.cat((*next_obs,*opp_next_obs),dim=2)
            mod_all_trgt_acs = torch.cat((*all_trgt_acs,*opp_all_trgt_acs),dim=2)
            # Target critic values
            trgt_vf_in = torch.cat((mod_next_obs, mod_all_trgt_acs), dim=2)
            if self.TD3: # TODO* For D4PG case, need mask with indices of the distributions whos distr_to_q(trgtQ1) < distr_to_q(trgtQ2)
                        # and build the combination of distr choosing the minimums
                trgt_Q1,trgt_Q2,_,_ = curr_agent.target_critic(trgt_vf_in)
                trgt_Q1 = trgt_Q1[1:].detach()
                trgt_Q2 = trgt_Q2[1:].detach()                
                if self.D4PG:
                    arg = torch.argmin(torch.stack((curr_agent.target_critic.distr_to_q(trgt_Q1).mean(),
                                    curr_agent.target_critic.distr_to_q(trgt_Q2).mean()),dim=0))
                    if not arg: 
                        trgt_Q = trgt_Q1
                    else:
                        trgt_Q = trgt_Q2
                else:
                    trgt_Q = torch.min(trgt_Q1,trgt_Q2)
            else:
                trgt_Q = curr_agent.target_critic(trgt_vf_in)
            if self.zero_critic:
                mod_acs = torch.cat((*[zero_params(a) for a in slice_acs],*[zero_params(a) for a in slice_opp_acs]),dim=2)
            else:
                mod_acs = torch.cat((*slice_acs,*slice_opp_acs),dim=2)
            mod_obs = torch.cat((*slice_obs,*slice_opp_obs),dim=2)
            # Actual critic values
            vf_in = torch.cat((mod_obs, mod_acs), dim=2)
            if self.TD3:
                actual_value_1, actual_value_2,_,_ = curr_agent.critic(vf_in)
                actual_value_1 = actual_value_1.view(-1,1)
                actual_value_2 = actual_value_2.view(-1,1)
            else:
                actual_value = curr_agent.critic(vf_in)
            if self.D4PG:
                    # Q1
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device)
                    if self.TD3:
                        prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                        prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                        # distribution distance function
                        vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                    else:
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                        trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                        gamma=self.gamma**self.n_steps,device=self.device) 
                        # distribution distance function
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
            else: # single critic value
                target_value = (1-self.beta)*(torch.cat([n.view(-1,1) for n in n_step_rews],dim=1).float().mean(dim=1).view(-1, 1) + (self.gamma**self.n_steps) *
                            trgt_Q.view(-1,1) * (1 - dones[agent_i].view(-1, 1))) + self.beta*(torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1)).view(-1,1)
                target_value.detach()
                if self.TD3: # handle double critic
                    
                    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                    prio = np.round(prio.cpu().numpy(),decimals=3)
                    prio = torch.tensor(prio,requires_grad=False)
                    vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
                else:
                    vf_loss = F.mse_loss(actual_value, target_value)
            vf_loss.backward() 
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
            curr_agent.critic_optimizer.step()
            self.repackage_hidden(h1)
            self.repackage_hidden(h2)
            self.repackage_hidden(h1_target)
            self.repackage_hidden(h2_target)
        # Train actor -----------------------
        if policy:

            obs = sorted_feats[0][0] #use features sorted by prox and stacked per agent along batch
            opp_obs = sorted_feats[0][1]
            acs = sorted_feats[0][2]
            opp_acs = sorted_feats[0][3] 
            curr_agent.policy_optimizer.zero_grad()
            
            self.zero_hidden(self.batch_size,actual=True,target=True,torch_device=self.torch_device)
            self.zero_hidden_policy(self.batch_size*nagents,torch_device=self.torch_device)
            #h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device)
            #self.set_hidden(h1, h2,actual=True,target=False,torch_device=self.torch_device)
            burnin_slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            burnin_slice_acs = list(map(lambda x: x[:lstm_burn_in], acs))
            burnin_slice_obs_opp = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            burnin_slice_acs_opp = list(map(lambda x: x[:lstm_burn_in], opp_acs))
            if self.zero_critic:
                burnin_slice_acs = [zero_params(a) for a in burnin_slice_acs]
                burnin_slice_acs_opp = [zero_params(a) for a in burnin_slice_acs_opp]
            
            burnin_mod_obs = torch.cat((*burnin_slice_obs, *burnin_slice_obs_opp), dim=2)
            burnin_mod_acs = torch.cat((*burnin_slice_acs, *burnin_slice_acs_opp), dim=2)
            burnin_critic_tensor = torch.cat((burnin_mod_obs,burnin_mod_acs), dim=2)
            # Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burnin_critic_tensor)
            # Run burn-in on policy to refresh hidden states
            burnin_policy_tensor = torch.cat(burnin_slice_obs,dim=1)
            _ = curr_agent.policy(burnin_policy_tensor) # burn in
            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs)) # Not used currently
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))
            stacked_slice_obs = torch.cat(slice_obs,dim=1)
    
            if self.I2A:
                print("no implementation")
                #curr_pol_out = curr_agent.policy(slice_obs)
            else:
                curr_pol_out = curr_agent.policy(stacked_slice_obs)

            team_pol_acs = torch.cat((gumbel_softmax(torch.log_softmax(curr_pol_out[:,:,:curr_agent.action_dim],dim=2), hard=True, device=self.torch_device,LSTM=True),curr_pol_out[:,:,curr_agent.action_dim:]),dim=2)
            if self.zero_critic:
                team_pol_acs = zero_params(team_pol_acs)
                slice_opp_acs = [zero_params(a) for a in slice_opp_acs]
                
            curr_pol_out_stacked = team_pol_acs
            offset = self.batch_size
            # recreate list of agents shape instead of stacked agent shape
            team_pol_acs = [team_pol_acs[:,(offset*i):(offset*(i+1)),:] for i in range(nagents)] 
            obs_vf_in = torch.cat((*slice_obs,*slice_opp_obs),dim=2)
            acs_vf_in = torch.cat((*team_pol_acs,*slice_opp_acs),dim=2)
            mod_vf_in = torch.cat((obs_vf_in, acs_vf_in), dim=2)
            # ------------------------------------------------------
            if self.D4PG:
                if self.data_parallel:
                    critic_out = curr_agent.critic.module.Q1(mod_vf_in)
                else:
                    critic_out,_ = curr_agent.critic.Q1(mod_vf_in)
                distr_q = curr_agent.critic.distr_to_q(critic_out)
                pol_loss = -distr_q.mean()
            else: # non-distributional
                if self.data_parallel:
                    pol_loss,_ = -curr_agent.critic.module.Q1(mod_vf_in).view(-1,1).mean() 
                else:
                    pol_loss,_ = curr_agent.critic.Q1(mod_vf_in)
                    pol_loss = -pol_loss.view(-1,1).mean() 
    
            if self.D4PG:
                reg_param = 5.0
            else:
                reg_param = 5.0

            param_reg = torch.clamp((curr_pol_out_stacked[:,:,curr_agent.action_dim:]**2)-torch.ones_like(curr_pol_out_stacked[:,:,curr_agent.action_dim:]),min=0.0).sum(dim=2).mean() # How much parameters exceed (-1,1) bound
            entropy_reg = (-torch.log_softmax(curr_pol_out[:,:,:curr_agent.action_dim],dim=2).sum(dim=2).mean() * 1e-3)/reg_param # regularize using log probabilities
            pol_loss += param_reg
            pol_loss += entropy_reg
            if self.I2A:
                pol_loss.backward(retain_graph = True)
            else:
                pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_optimizer.step()
            self.repackage_hidden(h1)
            self.repackage_hidden(h2)

                #print(time.time() - start,"update time")
        # I2A --------------------------------------
        if self.I2A and policy:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()

            labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            self.ws_onehot.zero_() # reset OH tensor
            self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
            agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch

            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew# + self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        if policy:
            if self.niter % 100 == 0:
                print("Team (%s) Agent (%i) Actor loss:" % (side,agent_i),pol_loss)

                if self.I2A:
                    print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                    print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)


                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4),
                                                                        'prime_loss': np.round(pol_prime_loss.item(),4),
                                                                        'em_loss': np.round(EM_loss.item(),4)},
                                                                        ignore_index=True)
                else:
                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4)},                    
                                                                        ignore_index=True)
            #-------
            #---------------------------------------------------------------------------------
        if side == 'team':
            self.team_count[agent_i] += 1
        else:
            self.opp_count[agent_i] += 1
        # ------------------------------------
        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)
        self.niter +=1
        #print(time.time() - start,"up")
        if critic:
            if self.niter % 100 == 0:
                self.critic_loss_logger = self.critic_loss_logger.append({'iteration':self.niter,
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)


        # return priorities
        if critic:
            if self.TD3:
                if self.D4PG:
                    return ((prob_dist_1.float().sum(dim=1) + prob_dist_2.float().sum(dim=1))/2.0).cpu()
                else:
                    return prio

            else:
                return prob_dist.sum(dim=1).cpu()
        else:
            return None

    def update_LSTM(self, team_sample=[], opp_sample=[], agent_i=0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,
                                        forward_pass=True,load_same_agent=False,critic=False,policy=True,session_path="",lstm_burn_in=40):
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
        count = self.team_count[agent_i]
        curr_agent = self.team_agents[agent_i]
        target_policies = self.team_target_policies
        opp_target_policies = self.opp_target_policies
        nagents = self.nagents_team
        policies = self.team_policies
        opp_policies = self.opp_policies
        obs, acs, rews, dones, MC_rews,n_step_rews,ws,rec_states,sorted_feats = team_sample 
        # sorted feats = [agent_0:[tobs,oobs,tacs,oacs],agent_1:[tobs,oobs,tacs,oacs]] sorted by proximity
        opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_,_ = opp_sample

        self.curr_agent_index = agent_i
        if load_same_agent:
            curr_agent = self.team_agents[0]

        # Train critic ------------------------
        if critic:
            curr_agent.critic_optimizer.zero_grad()

            self.zero_hidden(self.batch_size*nagents,actual=True,target=True,torch_device=self.torch_device)
            self.zero_hidden_policy(self.batch_size*nagents,torch_device=self.torch_device)

            #h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device) # uncomment when recurrent states are fixed
            #self.set_hidden(h1, h2,actual=True,target=True,torch_device=self.torch_device)

            
            # Stack each agents perspective into the batch
            if self.zero_critic:
                for i in range(nagents):
                    sorted_feats[i][2] = [zero_params(a) for a in sorted_feats[i][2]]
                    sorted_feats[i][3] = [zero_params(a) for a in sorted_feats[i][3]]

            obs = [torch.cat([sorted_feats[i][0][j] for i in range(nagents)],dim=1) for j in range(nagents)]  # use features sorted by prox and stacked per agent along batch
            opp_obs = [torch.cat([sorted_feats[i][1][j] for i in range(nagents)],dim=1) for j in range(nagents)]
            acs =  [torch.cat([sorted_feats[i][2][j] for i in range(nagents)],dim=1) for j in range(nagents)]
            opp_acs = [torch.cat([sorted_feats[i][3][j] for i in range(nagents)],dim=1) for j in range(nagents)]
            n_step_rews = [val.repeat(1,nagents,1) for val in n_step_rews]
            MC_rews = [val.repeat(1,nagents,1) for val in MC_rews]
            dones = [val.repeat(1,nagents,1) for val in dones]

            # obs = sorted_feats[0][0] #use features sorted by prox and stacked per agent along batch
            # opp_obs = sorted_feats[0][1]
            # acs = sorted_feats[0][2]
            # opp_acs = sorted_feats[0][3] 
            #n_step_rews = [val.repeat(1,nagents,1) for val in n_step_rews]
            #MC_rews = [val.repeat(1,nagents,1) for val in MC_rews]
            #dones = [val.repeat(1,nagents,1) for val in dones]

            # obs of first agent minus his last action concat with all of his teammates stamina concat with all agents acs
            critic_input = torch.cat((torch.cat((obs[0][:,:,:-8],torch.stack(([obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2),torch.cat(acs,dim=2),torch.cat(opp_acs,dim=2)),dim=2)
            burn_in_tensor = critic_input[:lstm_burn_in]
            # Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burn_in_tensor)
            _,_,h1_target,h2_target = curr_agent.target_critic(burn_in_tensor)

            # Run burn-in on target policy
            burnin_slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            [pi(nobs) for pi,nobs in zip(target_policies,burnin_slice_obs)]# burn in target
            # Run burn-in on opponent target policy
            burnin_opp_slice_obs = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            [pi(nobs) for pi,nobs in zip(opp_target_policies,burnin_opp_slice_obs)]# burn in opp target


            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs))
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))
            n_step_rews = list(map(lambda x: x[lstm_burn_in:], n_step_rews)) # extend dones and rewards using repeat
            MC_rews = list(map(lambda x: x[lstm_burn_in:], MC_rews))
            dones = list(map(lambda x: x[lstm_burn_in:], dones))
            next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), obs))
            opp_next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), opp_obs))
            start = time.time()

            if self.TD3:
                noise = processor(torch.randn_like(acs[0][lstm_burn_in-1:]),device=self.device,torch_device=self.torch_device) * self.TD3_noise
                if self.I2A:
                    team_pi_acs = [a + noise for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a + noise for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(opp_target_policies,opp_next_obs)]

                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=True),out[:,:,curr_agent.action_dim:]),dim=2) for out in team_pi_acs]

                opp_all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=True),out[:,:,curr_agent.action_dim:]),dim=2) for out in opp_pi_acs]
            else:
                if self.I2A:
                    team_pi_acs = [a for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs)) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs)) for pi, nobs in zip(opp_target_policies,opp_next_obs)]

                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in team_pi_acs]
                opp_all_trgt_acs =[torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in opp_pi_acs]

            if self.zero_critic:
                all_trgt_acs = [zero_params(a) for a in all_trgt_acs]
                opp_all_trgt_acs = [zero_params(a) for a in opp_all_trgt_acs]
                
            mod_next_obs = torch.cat((next_obs[0][:,:,:-8],torch.stack(([next_obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2)
            mod_all_trgt_acs = torch.cat((*all_trgt_acs,*opp_all_trgt_acs),dim=2)

            # Target critic values
            trgt_vf_in = torch.cat((mod_next_obs, mod_all_trgt_acs), dim=2)
            if self.TD3: # TODO* For D4PG case, need mask with indices of the distributions whos distr_to_q(trgtQ1) < distr_to_q(trgtQ2)
                        # and build the combination of distr choosing the minimums
                trgt_Q1,trgt_Q2,_,_ = curr_agent.target_critic(trgt_vf_in)
                trgt_Q1 = trgt_Q1[1:].detach()
                trgt_Q2 = trgt_Q2[1:].detach()                
                if self.D4PG:
                    arg = torch.argmin(torch.stack((curr_agent.target_critic.distr_to_q(trgt_Q1).mean(),
                                    curr_agent.target_critic.distr_to_q(trgt_Q2).mean()),dim=0))

                    if not arg: 
                        trgt_Q = trgt_Q1
                    else:
                        trgt_Q = trgt_Q2
                else:
                    trgt_Q = torch.min(trgt_Q1,trgt_Q2)
            else:
                trgt_Q = curr_agent.target_critic(trgt_vf_in)

            if self.zero_critic:
                mod_acs = torch.cat((*[zero_params(a) for a in slice_acs],*[zero_params(a) for a in slice_opp_acs]),dim=2)
            else:
                mod_acs = torch.cat((*slice_acs,*slice_opp_acs),dim=2)

            mod_obs = torch.cat((slice_obs[0][:,:,:-8],torch.stack(([slice_obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2)
            # Actual critic values
            vf_in = torch.cat((mod_obs, mod_acs), dim=2)
            if self.TD3:
                actual_value_1, actual_value_2,_,_ = curr_agent.critic(vf_in)
                actual_value_1 = actual_value_1.view(-1,1)
                actual_value_2 = actual_value_2.view(-1,1)
            else:
                actual_value = curr_agent.critic(vf_in)

            if self.D4PG:
                    # Q1
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution

                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device)

                    if self.TD3:
                        prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                        prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                        # distribution distance function
                        vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                    else:
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                        trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                        gamma=self.gamma**self.n_steps,device=self.device) 
                        # distribution distance function
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
            else: # single critic value
                target_value = (1-self.beta)*(torch.cat([n.view(-1,1) for n in n_step_rews],dim=1).float().mean(dim=1).view(-1, 1) + (self.gamma**self.n_steps) *
                            trgt_Q.view(-1,1) * (1 - dones[agent_i].view(-1, 1))) + self.beta*(torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1)).view(-1,1)
                target_value.detach()

                if self.TD3: # handle double critic
                    
                    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                    prio = np.round(prio.cpu().numpy(),decimals=3)
                    prio = torch.tensor(prio,requires_grad=False)
                    vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
                else:
                    vf_loss = F.mse_loss(actual_value, target_value)

            vf_loss.backward() 

            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
            curr_agent.critic_optimizer.step()
            self.repackage_hidden(h1)
            self.repackage_hidden(h2)
            self.repackage_hidden(h1_target)
            self.repackage_hidden(h2_target)



        # Train actor -----------------------
        if policy:
            curr_agent.policy_optimizer.zero_grad()
            
            self.zero_hidden(self.batch_size,actual=True,target=True,torch_device=self.torch_device)
            self.zero_hidden_policy(self.batch_size*nagents,torch_device=self.torch_device)
            #h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device)
            #self.set_hidden(h1, h2,actual=True,target=False,torch_device=self.torch_device)

            if self.zero_critic:
                for i in range(nagents):
                    sorted_feats[i][2] = [zero_params(a) for a in sorted_feats[i][2]]
                    sorted_feats[i][3] = [zero_params(a) for a in sorted_feats[i][3]]

            obs = sorted_feats[0][0]  # use features sorted by prox and stacked per agent along batch
            opp_obs =  sorted_feats[0][1]
            acs =  sorted_feats[0][2] 
            opp_acs = sorted_feats[0][3]

            critic_input = torch.cat((torch.cat((obs[0][:,:,:-8],torch.stack(([obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2),torch.cat(acs,dim=2),torch.cat(opp_acs,dim=2)),dim=2)
            burn_in_tensor = critic_input[:lstm_burn_in]

            # Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burn_in_tensor)

            burnin_slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))

            # Run burn-in on policy to refresh hidden states
            burnin_policy_tensor = torch.cat(burnin_slice_obs,dim=1)
            _ = curr_agent.policy(burnin_policy_tensor) # burn in


            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs)) # Not used currently
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))

            stacked_slice_obs = torch.cat(slice_obs,dim=1)

    
            if self.I2A:
                print("no implementation")
                #curr_pol_out = curr_agent.policy(slice_obs)
            else:
                curr_pol_out = curr_agent.policy(stacked_slice_obs)

            team_pol_acs = torch.cat((gumbel_softmax(torch.log_softmax(curr_pol_out[:,:,:curr_agent.action_dim],dim=2), hard=True, device=self.torch_device,LSTM=True),curr_pol_out[:,:,curr_agent.action_dim:]),dim=2)
            if self.zero_critic:
                team_pol_acs = zero_params(team_pol_acs)
                slice_opp_acs = [zero_params(a) for a in slice_opp_acs]

                
            curr_pol_out_stacked = team_pol_acs
            offset = self.batch_size
            # recreate list of agents shape instead of stacked agent shape
            team_pol_acs = [team_pol_acs[:,(offset*i):(offset*(i+1)),:] for i in range(nagents)] 



            obs_vf_in = torch.cat((slice_obs[0][:,:,:-8],torch.stack(([slice_obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2)
            acs_vf_in = torch.cat((*team_pol_acs,*slice_opp_acs),dim=2)
            mod_vf_in = torch.cat((obs_vf_in, acs_vf_in), dim=2)

            # ------------------------------------------------------
            if self.D4PG:
                if self.data_parallel:
                    critic_out = curr_agent.critic.module.Q1(mod_vf_in)
                else:
                    critic_out,_ = curr_agent.critic.Q1(mod_vf_in)
                distr_q = curr_agent.critic.distr_to_q(critic_out)
                pol_loss = -distr_q.mean()
            else: # non-distributional
                if self.data_parallel:
                    pol_loss,_ = -curr_agent.critic.module.Q1(mod_vf_in).view(-1,1).mean() 
                else:
                    pol_loss,_ = curr_agent.critic.Q1(mod_vf_in)
                    pol_loss = -pol_loss.view(-1,1).mean() 
    
            if self.D4PG:
                reg_param = 5.0
            else:
                reg_param = 10.0
            
            param_reg = torch.clamp((curr_pol_out_stacked[:,:,curr_agent.action_dim:]**2)-torch.ones_like(curr_pol_out_stacked[:,:,curr_agent.action_dim:]),min=0.0).sum(dim=2).mean() # How much parameters exceed (-1,1) bound
            entropy_reg = (-torch.log_softmax(curr_pol_out[:,:,:curr_agent.action_dim],dim=2).sum(dim=2).mean() * 1e-3)/reg_param # regularize using log probabilities
            pol_loss += param_reg
            pol_loss += entropy_reg
            if self.I2A:
                pol_loss.backward(retain_graph = True)
            else:
                pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_optimizer.step()
            self.repackage_hidden(h1)
            self.repackage_hidden(h2)

                #print(time.time() - start,"update time")
        # I2A --------------------------------------
        if self.I2A and policy:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            self.ws_onehot.zero_() # reset OH tensor
            self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
            agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch
            
            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew# + self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        if policy:
            if self.niter % 100 == 0:
                print("Team (%s) Agent (%i) Actor loss:" % (side,agent_i),pol_loss)

                if self.I2A:
                    print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                    print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)
    

                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4),
                                                                        'prime_loss': np.round(pol_prime_loss.item(),4),
                                                                        'em_loss': np.round(EM_loss.item(),4)},
                                                                        ignore_index=True)
                else:
                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4)},                    
                                                                        ignore_index=True)
            #-------
            #---------------------------------------------------------------------------------
        if side == 'team':
            self.team_count[agent_i] += 1
        else:
            self.opp_count[agent_i] += 1
        # ------------------------------------
        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)
        self.niter +=1
        #print(time.time() - start,"up")
        if critic:
            if self.niter % 100 == 0:
                self.critic_loss_logger = self.critic_loss_logger.append({'iteration':self.niter,
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)


        # return priorities
        if critic:
            if self.TD3:
                if self.D4PG:
                    return ((prob_dist_1.float().sum(dim=1) + prob_dist_2.float().sum(dim=1))/2.0).cpu()
                else:
                    return prio
                    
            else:
                return prob_dist.sum(dim=1).cpu()
        else:
            return None


    def update_centralized_critic_LSTM(self, team_sample=[], opp_sample=[], agent_i=0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,
                                        forward_pass=True,load_same_agent=False,critic=True,policy=True,session_path="",lstm_burn_in=40):
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
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            opp_target_policies = self.opp_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            obs, acs, rews, dones, MC_rews,n_step_rews,ws,rec_states = team_sample
            opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_ = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            opp_target_policies = self.team_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, dones,MC_rews,n_step_rews,ws,rec_states = opp_sample
            opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_ = team_sample

        self.curr_agent_index = agent_i
        if load_same_agent:
            curr_agent = self.team_agents[0]

        # Train critic ------------------------
        if critic:

            curr_agent.critic_optimizer.zero_grad()
            
            self.zero_hidden(self.batch_size,torch_device=self.torch_device)
            self.zero_hidden_policy(self.batch_size,torch_device=self.torch_device)
            h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device)
            self.set_hidden(h1, h2,actual=True,target=True,torch_device=self.torch_device)

            slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            slice_acs = list(map(lambda x: x[:lstm_burn_in], acs))
            slice_obs_opp = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            slice_acs_opp = list(map(lambda x: x[:lstm_burn_in], opp_acs))

            mod_obs = torch.cat((*slice_obs, *slice_obs_opp), dim=2)
            mod_acs = torch.cat((*slice_acs, *slice_acs_opp), dim=2)

            burn_in_tensor = torch.cat((mod_obs, mod_acs), dim=2)

            # Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burn_in_tensor)
            _,_,h1_target,h2_target = curr_agent.target_critic(burn_in_tensor)
            
            curr_agent.critic_optimizer.zero_grad()

            self.set_hidden(h1,h2,actual=True,target=False)
            self.set_hidden(h1_target,h2_target,actual=False,target=True)

            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs))
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))
            n_step_rews = list(map(lambda x: x[lstm_burn_in:], n_step_rews))
            MC_rews = list(map(lambda x: x[lstm_burn_in:], MC_rews))
            dones = list(map(lambda x: x[lstm_burn_in:], dones))
            next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), obs))
            opp_next_obs = list(map(lambda x: torch.cat((x[lstm_burn_in:],x[-1:, :, :]),dim=0), opp_obs))

            #print("time critic")
            #start = time.time()
            #with torch.no_grad():
            if self.TD3:
                noise = processor(torch.randn_like(acs[0][lstm_burn_in-1:]),device=self.device,torch_device=self.torch_device) * self.TD3_noise
                if self.I2A:
                    team_pi_acs = [a + noise for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a + noise for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs) + noise) for pi, nobs in zip(opp_target_policies,opp_next_obs)]

                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=self.LSTM),out[:,:,curr_agent.action_dim:]),dim=2) for out in team_pi_acs]

                opp_all_trgt_acs = [torch.cat(
                (onehot_from_logits(out[:,:,:curr_agent.action_dim], LSTM=self.LSTM),out[:,:,curr_agent.action_dim:]),dim=2) for out in opp_pi_acs]
            else:
                if self.I2A:
                    team_pi_acs = [a for a in target_policies[0](next_obs)] # get actions for all agents, add noise to each
                    opp_pi_acs = [a for a in opp_target_policies[0](opp_next_obs)] # get actions for all agents, add noise to each
                else:
                    team_pi_acs  = [(pi(nobs)) for pi, nobs in zip(target_policies,next_obs)]
                    opp_pi_acs  = [(pi(nobs)) for pi, nobs in zip(opp_target_policies,opp_next_obs)]

                all_trgt_acs = [torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in team_pi_acs]
                opp_all_trgt_acs =[torch.cat(
                    (onehot_from_logits(out[:,:curr_agent.action_dim]),out[:,curr_agent.action_dim:]),dim=1) for out in opp_pi_acs]

            if self.zero_critic:
                all_trgt_acs = [zero_params(a) for a in all_trgt_acs]
                opp_all_trgt_acs = [zero_params(a) for a in opp_all_trgt_acs]
                
            mod_next_obs = torch.cat((*next_obs,*opp_next_obs),dim=2)
            mod_all_trgt_acs = torch.cat((*all_trgt_acs,*opp_all_trgt_acs),dim=2)

            # Target critic values
            trgt_vf_in = torch.cat((mod_next_obs, mod_all_trgt_acs), dim=2)
            if self.TD3: # TODO* For D4PG case, need mask with indices of the distributions whos distr_to_q(trgtQ1) < distr_to_q(trgtQ2)
                        # and build the combination of distr choosing the minimums
                trgt_Q1,trgt_Q2,_,_ = curr_agent.target_critic(trgt_vf_in)
                trgt_Q1 = trgt_Q1[1:]
                trgt_Q2 = trgt_Q2[1:]
                
                if self.D4PG:
                    arg = torch.argmin(torch.stack((curr_agent.target_critic.distr_to_q(trgt_Q1).mean(),
                                    curr_agent.target_critic.distr_to_q(trgt_Q2).mean()),dim=0))

                    if not arg: 
                        trgt_Q = trgt_Q1
                    else:
                        trgt_Q = trgt_Q2
                else:
                    trgt_Q = torch.min(trgt_Q1,trgt_Q2)
            else:
                trgt_Q = curr_agent.target_critic(trgt_vf_in)

            if self.zero_critic:
                mod_acs = torch.cat((*[zero_params(a) for a in slice_opp_acs],*[zero_params(a) for a in slice_acs]),dim=2)
            else:
                mod_acs = torch.cat((*slice_acs,*slice_opp_acs),dim=2)

            mod_obs = torch.cat((*slice_obs,*slice_opp_obs),dim=2)
            # Actual critic values
            vf_in = torch.cat((mod_obs, mod_acs), dim=2)
            if self.TD3:
                actual_value_1, actual_value_2,_,_ = curr_agent.critic(vf_in)
                actual_value_1 = actual_value_1.view(-1,1)
                actual_value_2 = actual_value_2.view(-1,1)
            else:
                actual_value = curr_agent.critic(vf_in)

            if self.D4PG:
                    # Q1
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution

                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device)

                    if self.TD3:
                        prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                        prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                        # distribution distance function
                        vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                    else:
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                        trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                        gamma=self.gamma**self.n_steps,device=self.device) 
                        # distribution distance function
                        prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                        vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
            else: # single critic value
                target_value = (1-self.beta)*(torch.cat([n.view(-1,1) for n in n_step_rews],dim=1).float().mean(dim=1).view(-1, 1) + (self.gamma**self.n_steps) *
                            trgt_Q.view(-1,1) * (1 - dones[agent_i].view(-1, 1))) + self.beta*(torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1)).view(-1,1)
                target_value.detach()

                if self.TD3: # handle double critic
                    
                    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                    prio = np.round(prio.cpu().numpy(),decimals=3)
                    prio = torch.tensor(prio,requires_grad=False)
                    vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
                else:
                    vf_loss = F.mse_loss(actual_value, target_value)

            vf_loss.backward(retrain_graph=True) 
            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
            curr_agent.critic_optimizer.step()


        # Train actor -----------------------
        if policy:
            curr_agent.policy_optimizer.zero_grad()
            self.zero_hidden(self.batch_size,actual=True,target=True,torch_device=self.torch_device)
            h1, h2 = self.cast_hidden(rec_states,torch_device=self.torch_device)
            self.set_hidden(h1, h2,actual=True,target=False,torch_device=self.torch_device)

            slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            slice_obs_opp = list(map(lambda x: x[:lstm_burn_in], opp_obs))
            slice_acs = list(map(lambda x: x[:lstm_burn_in], acs))
            slice_acs_opp = list(map(lambda x: x[:lstm_burn_in], opp_acs))

            mod_obs = torch.cat((*slice_obs, *slice_obs_opp), dim=2)
            mod_acs = torch.cat((*slice_acs, *slice_acs_opp), dim=2)

            burn_in_tensor = torch.cat((mod_obs, mod_acs), dim=2)

            # Run burn-in on critic to refresh hidden states
            _,_,h1,h2 = curr_agent.critic(burn_in_tensor)
            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs)) # Not used currently
            slice_opp_obs = list(map(lambda x: x[lstm_burn_in:], opp_obs))
            slice_opp_acs = list(map(lambda x: x[lstm_burn_in:], opp_acs))

                    
            #print("time actor")
            team_pol_acs = []
    
            if self.I2A:
                curr_pol_out = curr_agent.policy(slice_obs)
            else:
                curr_pol_out = [curr_agent.policy(slice_obs[ag]) for ag in range(nagents)]

            team_pol_acs = [torch.cat((gumbel_softmax(c[:,:,:curr_agent.action_dim], hard=True, device=self.torch_device,LSTM=self.LSTM),c[:,:,curr_agent.action_dim:]),dim=2) for c in curr_pol_out]
            curr_pol_out_stacked = torch.cat(curr_pol_out,dim=1)


            obs_vf_in = torch.cat((*slice_obs,*slice_obs_opp),dim=2)
            acs_vf_in = torch.cat((*team_pol_acs,*slice_acs_opp),dim=2)
            mod_vf_in = torch.cat((obs_vf_in, acs_vf_in), dim=2)

            # ------------------------------------------------------
            if self.D4PG:
                if self.data_parallel:
                    critic_out = curr_agent.critic.module.Q1(mod_vf_in)
                else:
                    critic_out = curr_agent.critic.Q1(mod_vf_in)
                distr_q = curr_agent.critic.distr_to_q(critic_out)
                pol_loss = -distr_q.mean()
            else: # non-distributional
                if self.data_parallel:
                    pol_loss = -curr_agent.critic.module.Q1(mod_vf_in).view(-1,1).mean() 
                else:
                    pol_loss = -curr_agent.critic.Q1(mod_vf_in).view(-1,1).mean() 
    
            if self.D4PG:
                reg_param = 5.0
            else:
                reg_param = 2.5
            
            param_reg = torch.clamp((curr_pol_out_stacked[:,:,curr_agent.action_dim:]**2)-torch.ones_like(curr_pol_out_stacked[:,:,curr_agent.action_dim:]),min=0.0).sum(dim=2).mean()
            entropy_reg = (-torch.log_softmax(curr_pol_out_stacked[:,:,:curr_agent.action_dim],dim=2).sum(dim=2).mean() * 1e-3)/reg_param # regularize using log probabilities
            pol_loss += param_reg
            pol_loss += entropy_reg
            if self.I2A:
                pol_loss.backward(retain_graph = True)
            else:
                pol_loss.backward()
            if parallel:
                average_gradients(curr_agent.policy)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_optimizer.step()

                #print(time.time() - start,"update time")
        # I2A --------------------------------------
        if self.I2A and policy:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            self.ws_onehot.zero_() # reset OH tensor
            self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
            agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch
            
            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew# + self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        if policy:
            if self.niter % 100 == 0:
                print("Team (%s) Agent (%i) Actor loss:" % (side,agent_i),pol_loss)

                if self.I2A:
                    print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                    print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)
    

                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4),
                                                                        'prime_loss': np.round(pol_prime_loss.item(),4),
                                                                        'em_loss': np.round(EM_loss.item(),4)},
                                                                        ignore_index=True)
                else:
                    self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor_loss': np.round(pol_loss.item(),4)},                    
                                                                        ignore_index=True)
            #-------
            #---------------------------------------------------------------------------------
        if side == 'team':
            self.team_count[agent_i] += 1
        else:
            self.opp_count[agent_i] += 1
        # ------------------------------------
        # if logger is not None:
        #     logger.add_scalars('agent%i/losses' % agent_i,
        #                        {'vf_loss': vf_loss,
        #                         'pol_loss': pol_loss},
        #                        self.niter)
        self.niter +=1
        #print(time.time() - start,"up")
        if critic:
            if self.niter % 100 == 0:
                self.critic_loss_logger = self.critic_loss_logger.append({'iteration':self.niter,
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)


        # return priorities
        if critic:
            if self.TD3:
                if self.D4PG:
                    return ((prob_dist_1.float().sum(dim=1) + prob_dist_2.float().sum(dim=1))/2.0).cpu()
                else:
                    return prio
                    
            else:
                return prob_dist.sum(dim=1).cpu()
        else:
            return None

    def inject(self,grad):
        new_grad = grad.clone()
        new_grad = self.invert(new_grad,self.param_dim)
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
    def invert(self,grad,num_params):

        agent_offset = self.nagents_team - 1 - self.curr_agent_index # since we are going backward we need to find the
                                                                     # position of the current agent going backwards
        index_offset = self.num_out_pol*agent_offset

        '''for sample in range(grad.shape[0]): # batch size
            for index in range(num_params):
            # last 5 are the params
                if grad[sample][-1 - index - index_offset] < 0:
                    grad[sample][-1 - index - index_offset] *= ((1.0-self.curr_pol_out[sample][-1 - index])/(1-(-1))) # scale
                else:
                    grad[sample][-1 - index - index_offset] *= ((self.curr_pol_out[sample][-1 - index]-(-1.0))/(1-(-1)))
        for sample in range(grad.shape[0]): # batch size
            # inverts gradients of discrete actions
            for index in range(3):
            # last 5 are the params
                if grad[sample][-1 - num_params - index - index_offset] < 0:
                    grad[sample][-1 - num_params - index - index_offset] *= ((1.0-self.curr_pol_out[sample][-1 - num_params -index])/(1-(-1))) # scale
                else:
                    grad[sample][-1 - num_params - index] *= ((self.curr_pol_out[sample][-1 - num_params - index]-(-1.0))/(1-(-1)))'''

        end_index = -index_offset
        if agent_offset == 0:
            end_index = len(grad[0])
        # last 5 are the params
        neg = grad[:,-self.num_out_pol-index_offset:end_index] < 0
        pos = grad[:,-self.num_out_pol-index_offset:end_index] >= 0
        neg = neg.float()
        pos = pos.float()
        grad_pos = grad[:, - self.num_out_pol- index_offset:end_index] * pos * (self.curr_pol_out[:, - self.num_out_pol:]-(-1.0))/(1-(-1))
        grad_neg = grad[:, - self.num_out_pol- index_offset:end_index] * neg * (1.0-self.curr_pol_out[:, - self.num_out_pol:])/(1-(-1))
        grad[:,-self.num_out_pol-index_offset:end_index] = grad_pos + grad_neg


        return grad
 
    def update_hard_critic(self):
        for a in self.team_agents:
            hard_update(a.target_critic, a.critic)

        for a in self.opp_agents:
            hard_update(a.target_critic, a.critic)

             
    def update_hard_policy(self):
        for a in self.team_agents:
            hard_update(a.target_policy, a.policy)

        for a in self.opp_agents:
            hard_update(a.target_policy, a.policy)

    def update_agent_hard_policy(self,agentID):
            hard_update(self.team_agents[agentID].target_policy, self.team_agents[agentID].policy)

    def update_agent_hard_critic(self,agentID):
            hard_update(self.team_agents[agentID].target_critic, self.team_agents[agentID].critic)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.team_agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)

        #for a in self.opp_agents:
        #    soft_update(a.target_critic, a.critic, self.tau)
        #    soft_update(a.target_policy, a.policy, self.tau)

        self.niter += 1

    def update_agent_targets(self,agentID,number_of_updates):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.team_agents[agentID].target_critic, self.team_agents[agentID].critic, self.tau*number_of_updates)
        soft_update(self.team_agents[agentID].target_policy, self.team_agents[agentID].policy, self.tau*number_of_updates)

        #for a in self.opp_agents:
        #    soft_update(a.target_critic, a.critic, self.tau)
        #    soft_update(a.target_policy, a.policy, self.tau)

        self.niter += 1

    def update_agent_actor(self,agentID,number_of_updates):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.team_agents[agentID].target_policy, self.team_agents[agentID].policy, self.tau*number_of_updates)

        #for a in self.opp_agents:
        #    soft_update(a.target_critic, a.critic, self.tau)
        #    soft_update(a.target_policy, a.policy, self.tau)

        self.niter += 1

    def update_agent_critic(self,agentID,number_of_updates):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.team_agents[agentID].target_critic, self.team_agents[agentID].critic, self.tau*number_of_updates)

        #for a in self.opp_agents:
        #    soft_update(a.target_critic, a.critic, self.tau)
        #    soft_update(a.target_policy, a.policy, self.tau)

        self.niter += 1

    def prep_policy(self,device='cuda',torch_device=torch.device('cuda:0')):
        for a in self.team_agents:
            a.policy.train()
            #a.reducer.train()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()

        for a in self.opp_agents:
            a.policy.train()
            #a.reducer.train()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()

        if device == 'cuda':
            fn = lambda x: x.to(torch_device)
        else:
            fn = lambda x: x.cpu()
            
        for a in self.team_agents:
            a.policy = fn(a.policy)
            #a.reducer = fn(a.reducer)
            if self.I2A:    
                a.policy_prime = fn(a.policy_prime)
                a.EM  = a.EM = fn(a.EM)

        for a in self.opp_agents:
            a.policy = fn(a.policy)
            #a.reducer = fn(a.reducer)
            if self.I2A:    
                a.policy_prime = fn(a.policy_prime)
                a.EM  = a.EM = fn(a.EM)
            
    def prep_training(self, device='gpu',only_policy=False,torch_device=torch.device('cuda:0')):
        for a in self.team_agents:
            a.policy.train()
            #a.reducer.train()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()
            if not only_policy:
                a.critic.train()
                a.target_policy.train()
                a.target_critic.train()
        
        for a in self.opp_agents:
            a.policy.train()
            #a.reducer.train()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()
            if not only_policy:
                a.critic.train()
                a.target_policy.train()
                a.target_critic.train()

        if device == 'cuda':
            fn = lambda x: x.to(torch_device)
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.team_agents:
                a.policy = fn(a.policy)
                #a.reducer = fn(a.reducer)
                if self.I2A:
                    a.policy_prime = fn(a.policy_prime)
                    a.EM  = a.EM = fn(a.EM)
            for a in self.opp_agents:
                a.policy = fn(a.policy)
                #a.reducer = fn(a.reducer)
                if self.I2A:
                    a.policy_prime = fn(a.policy_prime)
                    a.EM  = a.EM = fn(a.EM)
            self.pol_dev = device
        if not only_policy:
            #if not self.critic_dev == device:
            for a in self.team_agents:
                a.critic = fn(a.critic)
            for a in self.opp_agents:
                a.critic = fn(a.critic)
            self.critic_dev = device

            if not self.trgt_pol_dev == device:
                for a in self.team_agents:
                    a.target_policy = fn(a.target_policy)
                for a in self.opp_agents:
                    a.target_policy = fn(a.target_policy)
                self.trgt_pol_dev = device

            if not self.trgt_critic_dev == device:
                for a in self.team_agents:
                    a.target_critic = fn(a.target_critic)
                for a in self.opp_agents:
                    a.target_critic = fn(a.target_critic)
                self.trgt_critic_dev = device
            if self.I2A:

                if not self.EM_dev == device:
                    for a in self.team_agents:
                        a.EM = fn(a.EM)
                    self.EM_dev = device
                    for a in self.opp_agents:
                        a.EM = fn(a.EM)
                    self.EM_dev = device
                if not self.prime_dev == device:
                    for a in self.team_agents:
                        a.policy_prime = fn(a.policy_prime)
                    for a in self.opp_agents:
                        a.policy_prime = fn(a.policy_prime)
                    self.prime_dev = device
                if not self.imagination_pol_dev == device:
                    for a in self.team_agents:
                        a.imagination_policy = fn(a.imagination_policy)
                    for a in self.opp_agents:
                        a.imagination_policy = fn(a.imagination_policy)
                    self.imagination_pol_dev = device

    def prep_policy_rollout(self, device='gpu',only_policy=False,torch_device=torch.device('cuda:0')):
        for a in self.team_agents:
            a.policy.eval()
            #a.reducer.eval()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()

        for a in self.opp_agents:
            a.policy.train()
            #a.reducer.train()
            if self.I2A:
                a.policy_prime.train()
                a.EM.train()

        if device == 'cuda':
            fn = lambda x: x.to(torch_device)
        else:
            fn = lambda x: x.cpu()
            
        for a in self.team_agents:
            a.policy = fn(a.policy)
            #a.reducer = fn(a.reducer)
            if self.I2A:    
                a.policy_prime = fn(a.policy_prime)
                a.EM  = a.EM = fn(a.EM)

        for a in self.opp_agents:
            a.policy = fn(a.policy)
            #a.reducer = fn(a.reducer)
            if self.I2A:    
                a.policy_prime = fn(a.policy_prime)
                a.EM  = a.EM = fn(a.EM)
            
    
    #Needs to be tested
    def save_actor(self, filename):
        """
        Save trained parameters of all agent's actor network into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'actor_params': [a.get_actor_params() for a in (self.team_agents+self.opp_agents)]}
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

# ----------------------------
# - Pretraining Functions ----
    
    def pretrain_critic_LSTM(self, team_sample=[], opp_sample =[], agent_i = 0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,forward_pass=True,load_same_agent=False,session_path="",lstm_burn_in=40):
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
        if self.niter % 1000:
            self.critic_loss_logger.to_csv(session_path + 'critic_loss.csv')
    
        start = time.time()

        #start = time.time()
        # rews = 1-step, cum-rews = n-step
        count = self.team_count[agent_i]
        curr_agent = self.team_agents[agent_i]
        nagents = self.nagents_team
        policies = self.team_policies
        opp_policies = self.opp_policies
        obs, acs, rews, dones, MC_rews,n_step_rews,ws,rec_states,sorted_feats = team_sample 
        # sorted feats = [agent_0:[tobs,oobs,tacs,oacs],agent_1:[tobs,oobs,tacs,oacs]] sorted by proximity
        opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_,_ = opp_sample

        self.curr_agent_index = agent_i


        # Train critic ------------------------
        curr_agent.critic_optimizer.zero_grad()
        if load_same_agent:
            curr_agent = self.team_agents[0]
        
        self.zero_hidden(self.batch_size*nagents,actual=True,target=True,torch_device=self.torch_device)
        self.zero_hidden_policy(self.batch_size*nagents,torch_device=self.torch_device)

        if self.zero_critic:
            for i in range(nagents):
                sorted_feats[i][2] = [zero_params(a) for a in sorted_feats[i][2]]
                sorted_feats[i][3] = [zero_params(a) for a in sorted_feats[i][3]]


        obs = [torch.cat([sorted_feats[i][0][j] for i in range(nagents)],dim=1) for j in range(nagents)]  # use features sorted by prox and stacked per agent along batch
        opp_obs = [torch.cat([sorted_feats[i][1][j] for i in range(nagents)],dim=1) for j in range(nagents)]
        acs =  [torch.cat([sorted_feats[i][2][j] for i in range(nagents)],dim=1) for j in range(nagents)]
        opp_acs = [torch.cat([sorted_feats[i][3][j] for i in range(nagents)],dim=1) for j in range(nagents)]
        n_step_rews = [val.repeat(1,nagents,1) for val in n_step_rews]
        MC_rews = [val.repeat(1,nagents,1) for val in MC_rews]
        dones = [val.repeat(1,nagents,1) for val in dones]

        # Add TD3 Noise to actions
        noise = processor(torch.randn_like(acs[0]),device=self.device,torch_device=self.torch_device) * self.TD3_noise
        acs =  [a + noise for a in acs]
        opp_acs = [a + noise for a in opp_acs]  
        mod_obs = torch.cat((obs[0][:,:,:-8],torch.stack(([obs[1 + i][:,:,26] for i in range (nagents-1)]),dim=2)),dim=2)
        mod_acs = torch.cat((*acs,*opp_acs),dim=2)
        
        # Actual critic values
        vf_in = torch.cat((mod_obs, mod_acs), dim=2)
        if self.TD3:
            actual_value_1, actual_value_2,h1,h2 = curr_agent.critic(vf_in)
            actual_value_1 = actual_value_1.view(-1,1)
            actual_value_2 = actual_value_2.view(-1,1)
        else:
            print("no implementation for single Q critic")
        
        if self.D4PG:
                # Q1
                trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution

                trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                gamma=self.gamma**self.n_steps,device=self.device)

                if self.TD3:
                    prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                    prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                    # distribution distance function
                    vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                else:
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device) 
                    # distribution distance function
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
        else: # single critic value
            target_value = torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1).view(-1,1)
            target_value.detach()
            if self.TD3: # handle double critic
                #with torch.no_grad():
                #    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                #    prio = np.round(prio.cpu().numpy(),decimals=3)
                #    prio = torch.tensor(prio,requires_grad=False)
                vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
            else:
                vf_loss = F.mse_loss(actual_value, target_value)

            vf_loss.backward() 

            if parallel:
                average_gradients(curr_agent.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
            curr_agent.critic_optimizer.step()
            self.repackage_hidden(h1)
            self.repackage_hidden(h2)
            self.niter += 1
            if self.niter % 100 == 0:
                self.critic_loss_logger = self.critic_loss_logger.append({'iteration':self.niter,
                                                                            'critic': np.round(vf_loss.item(),4)},
                                                                            ignore_index=True)
                print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)

            return 0





   
    def pretrain_prime(self, sample, agent_i,side='team', parallel=False, logger=None):
        obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = sample
        if side == 'team':
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
        else:
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
        zero_values = False
        reducer = curr_agent.reducer
        obs = [reducer.reduce(o) for o in obs]
        next_obs = [reducer.reduce(no) for no in next_obs]

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy_prime(obs[agent_i]) # uses gumbel across the actions

            self.curr_pol_out = curr_pol_out.clone() # for inverting action space

        curr_agent.policy_prime_optimizer.zero_grad()
        
        pol_out_actions = curr_pol_out[:,:curr_agent.action_dim].float()
        actual_out_actions = Variable(torch.stack(acs)[agent_i],requires_grad=True).float()[:,:curr_agent.action_dim]
        pol_out_params = curr_pol_out[:,curr_agent.action_dim:]
        actual_out_params = Variable(torch.stack(acs)[agent_i],requires_grad=True)[:,curr_agent.action_dim:]

        target_classes = torch.argmax(actual_out_actions,dim=1) # categorical integer for predicted class

        #MSE =np.sum([F.mse_loss(estimation[self.discrete_param_indices(target_class)],actual[self.discrete_param_indices(target_class)]) for estimation,actual,target_class in zip(pol_out_params,actual_out_params, target_classes)])/(1.0*self.batch_size)

        MSE = F.mse_loss(pol_out_params,actual_out_params)
        #pol_prime_loss = MSE + CELoss(pol_out_actions,target_classes)
        pol_prime_loss = MSE + F.mse_loss(pol_out_actions,actual_out_actions)
        #pol_loss += (curr_pol_out[:curr_agent.action_dim]**2).mean() * 1e-2 # regularize size of action
        pol_prime_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy_prime)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_prime_optimizer.step()

        if self.niter % 100 == 0:
            print("Team (%s) Policy Prime Loss" % side,pol_prime_loss)
        
   


    def pretrain_critic_MC(self, team_sample=[], opp_sample =[], agent_i = 0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,forward_pass=True,load_same_agent=False,session_path=""):
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
        if self.niter % 1000:
            self.critic_loss_logger.to_csv(session_path + 'critic_loss.csv')
    
        start = time.time()

        #start = time.time()
        # rews = 1-step, cum-rews = n-step
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = team_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]

            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = opp_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = team_sample

        self.curr_agent_index = agent_i
        if self.preprocess:
            reducer = curr_agent.reducer
            obs = [reducer.reduce(o) for o in obs]
            next_obs = [reducer.reduce(no) for no in next_obs]
            opp_obs = [reducer.reduce(oo) for oo in opp_obs]
            opp_next_obs = [reducer.reduce(ono) for ono in opp_next_obs]
        # Train critic ------------------------
        curr_agent.critic_optimizer.zero_grad()
        if load_same_agent:
            curr_agent = self.team_agents[0]
        #print("time critic")
        #start = time.time()
        #with torch.no_grad():

        
        
        # Add TD3 Noise to actions
        noise = processor(torch.randn_like(acs[0]),device=self.device,torch_device=self.torch_device) * self.TD3_noise * 3.0
        acs =  [a + noise for a in acs]
        opp_acs = [a + noise for a in opp_acs]  
        mod_obs = torch.cat((*opp_obs,*obs),dim=1)
        mod_acs = torch.cat((*opp_acs,*acs),dim=1)
        if self.zero_critic:
            mod_acs = torch.cat((*[zero_params(a) for a in opp_acs],*[zero_params(a) for a in acs]),dim=1)
        # Actual critic values
        vf_in = torch.cat((mod_obs, mod_acs), dim=1)
        if self.TD3:
            actual_value_1, actual_value_2 = curr_agent.critic(vf_in)
        else:
            actual_value = curr_agent.critic(vf_in)
        
        if self.D4PG:
                # Q1
                trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution

                trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                gamma=self.gamma**self.n_steps,device=self.device)

                if self.TD3:
                    prob_dist_1 = -F.log_softmax(actual_value_1,dim=1) * trgt_vf_distr_proj # Q1
                    prob_dist_2 = -F.log_softmax(actual_value_2,dim=1) * trgt_vf_distr_proj # Q2
                    # distribution distance function
                    vf_loss = prob_dist_1.sum(dim=1).mean() + prob_dist_2.sum(dim=1).mean() # critic loss based on distribution distance
                else:
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    trgt_vf_distr = F.softmax(trgt_Q,dim=1) # critic distribution
                    trgt_vf_distr_proj = distr_projection(self,trgt_vf_distr,n_step_rews[agent_i],dones[agent_i],MC_rews[agent_i],
                                                    gamma=self.gamma**self.n_steps,device=self.device) 
                    # distribution distance function
                    prob_dist = -F.log_softmax(actual_value,dim=1) * trgt_vf_distr_proj
                    vf_loss = prob_dist.sum(dim=1).mean() # critic loss based on distribution distance
        else: # single critic value
            target_value = self.beta*(torch.cat([mc.view(-1,1) for mc in MC_rews],dim=1).float().mean(dim=1)).view(-1,1)
            target_value.detach()
            if self.TD3: # handle double critic
                with torch.no_grad():
                    prio = ((actual_value_1 - target_value)**2 + (actual_value_2-target_value)**2).squeeze().detach()/2.0
                    prio = np.round(prio.cpu().numpy(),decimals=3)
                    prio = torch.tensor(prio,requires_grad=False)
                vf_loss = F.mse_loss(actual_value_1, target_value) + F.mse_loss(actual_value_2,target_value)
            else:
                vf_loss = F.mse_loss(actual_value, target_value)
                        

        #vf_loss.backward()
        vf_loss.backward(retain_graph=False) 
        
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
        curr_agent.critic_optimizer.step()
        self.niter +=1
        #print(time.time() - start,"up")
        if self.niter % 100 == 0:
            self.critic_loss_logger = self.critic_loss_logger.append({                'iteration':self.niter,
                                                                        'critic': np.round(vf_loss.item(),4)},
                                                                        ignore_index=True)
            print("Team (%s) Agent(%i) Q loss" % (side, agent_i),vf_loss)

    
    # return priorities
        if self.TD3:
            if self.D4PG:
                return ((prob_dist_1.float().sum(dim=1) + prob_dist_2.float().sum(dim=1))/2.0).cpu()
            else:
                return prio     
        else:
            return prob_dist.sum(dim=1).cpu()


   
    def pretrain_prime(self, sample, agent_i,side='team', parallel=False, logger=None):
        obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = sample
        if side == 'team':
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
        else:
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
        zero_values = False
        reducer = curr_agent.reducer
        obs = [reducer.reduce(o) for o in obs]
        next_obs = [reducer.reduce(no) for no in next_obs]

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy_prime(obs[agent_i]) # uses gumbel across the actions

            self.curr_pol_out = curr_pol_out.clone() # for inverting action space

        curr_agent.policy_prime_optimizer.zero_grad()
        
        pol_out_actions = curr_pol_out[:,:curr_agent.action_dim].float()
        actual_out_actions = Variable(torch.stack(acs)[agent_i],requires_grad=True).float()[:,:curr_agent.action_dim]
        pol_out_params = curr_pol_out[:,curr_agent.action_dim:]
        actual_out_params = Variable(torch.stack(acs)[agent_i],requires_grad=True)[:,curr_agent.action_dim:]

        target_classes = torch.argmax(actual_out_actions,dim=1) # categorical integer for predicted class

        #MSE =np.sum([F.mse_loss(estimation[self.discrete_param_indices(target_class)],actual[self.discrete_param_indices(target_class)]) for estimation,actual,target_class in zip(pol_out_params,actual_out_params, target_classes)])/(1.0*self.batch_size)

        MSE = F.mse_loss(pol_out_params,actual_out_params)
        #pol_prime_loss = MSE + CELoss(pol_out_actions,target_classes)
        pol_prime_loss = MSE + F.mse_loss(pol_out_actions,actual_out_actions)
        #pol_loss += (curr_pol_out[:curr_agent.action_dim]**2).mean() * 1e-2 # regularize size of action
        pol_prime_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy_prime)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_prime_optimizer.step()

        if self.niter % 100 == 0:
            print("Team (%s) Policy Prime Loss" % side,pol_prime_loss)
        
   


        
    def update_prime(self, sample, agent_i, parallel=False, logger=None):
        obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = sample
        if side == 'team':
            curr_agent = self.team_agents[agent_i]
        else:
            curr_agent = self.opp_agents[agent_i]
            
       
        # Update policy prime
        curr_agent.policy_prime_optimizer.zero_grad()
        curr_pol_out = curr_agent.policy(obs[agent_i])
        # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
        pol_prime_out = curr_agent.policy_prime(obs[agent_i]) # uses gumbel across the actions
        pol_prime_out_actions = pol_prime_out[:,:curr_agent.action_dim].float()
        pol_prime_out_params = pol_prime_out[:,curr_agent.action_dim:]
        pol_out_actions = curr_pol_out[:,:curr_agent.action_dim].float()
        pol_out_params = curr_pol_out[:,curr_agent.action_dim:]
        target_classes = torch.argmax(pol_out_actions,dim=1) # categorical integer for predicted class
        MSE =np.sum([F.mse_loss(prime[self.discrete_param_indices(target_class)],current[self.discrete_param_indices(target_class)]) for prime,current,target_class in zip(pol_prime_out_params,pol_out_params, target_classes)])/(1.0*self.batch_size)
        #pol_prime_loss = MSE + CELoss(pol_out_actions,target_classes)
        pol_prime_loss = MSE + F.mse_loss(pol_prime_out_actions,pol_out_actions)
        pol_prime_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy_prime)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_prime_optimizer.step()

        if self.niter % 100 == 0:
            print("Team (%s) Policy Prime Loss" % side,pol_prime_loss)
        self.niter += 1
        
        
                                          
    # pretrain a policy from another team to use in imagination
                               
                                          
                                          
    def update_EM(self, sample, agent_i, side='team',parallel=False, logger=None):
        """
        Update parameters of Environment Model based on sample from replay buffer
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
        obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = sample
        if side == 'team':
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
        else:
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
            
        # Train Environment Model -----------------------------------------------------          
        if self.preprocess:
            curr_agent.EM_optimizer.zero_grad()
            reducer = curr_agent.reducer
            obs = [reducer.reduce(o) for o in obs]
            next_obs = [reducer.reduce(no) for no in next_obs]

        labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
        self.ws_onehot.zero_() # reset OH tensor
        self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
        #EM_in = torch.cat((*obs, *acs),dim=1) this is centralized
        EM_in = torch.cat((obs[agent_i],acs[agent_i]),dim=1)
        est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
        actual_obs_diff = next_obs[agent_i] - obs[agent_i]
        actual_rews = rews[agent_i].view(-1,1)
        actual_ws = self.ws_onehot
        loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
        loss_rew = F.mse_loss(est_rews, actual_rews)
        loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
        EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew + self.ws_weight * loss_ws
        EM_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
        curr_agent.EM_optimizer.step()

        if self.niter % 100 == 0:
            print("Team (%s) EM Loss:" % side,np.round(EM_loss.item(),2),"{Obs Loss:",np.round(loss_obs.item(),2),",","Rew Loss:",np.round(loss_rew.item(),2),",","WS Loss:",np.round(loss_ws.item(),2),"}")
        


    def pretrain_actor(self, team_sample=[], opp_sample =[], agent_i = 0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,forward_pass=True,load_same_agent=False,session_path=""):
        """
        Update parameters of actor based on sample from replay buffer for policy imitation (fits policy to observed actions)
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
        if self.niter % 1000:
            self.policy_loss_logger.to_csv(session_path + 'actor_loss.csv')

        start = time.time()

        #start = time.time()
        # rews = 1-step, cum-rews = n-step
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = team_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]

            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = opp_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = team_sample

        self.curr_agent_index = agent_i
        if self.preprocess:
            reducer = curr_agent.reducer
            obs = [reducer.reduce(o) for o in obs]
            next_obs = [reducer.reduce(no) for no in next_obs]
            opp_obs = [reducer.reduce(oo) for oo in opp_obs]
            opp_next_obs = [reducer.reduce(ono) for ono in opp_next_obs]
        if self.I2A:
            curr_pol_out = curr_agent.policy(obs)
        else:
            curr_pol_out = [curr_agent.policy(obs[ag]) for ag in range(nagents)]
        
        
        curr_pol_out_stacked = torch.cat(curr_pol_out,dim=0)
        
        #curr_pol_out = curr_agent.policy(all_obs)
        curr_agent.policy_optimizer.zero_grad()
        
        all_acs = torch.cat(acs,dim=0)
        
        pol_out_actions = torch.softmax(curr_pol_out_stacked[:,:curr_agent.action_dim],dim=1).float()
        actual_out_actions = Variable(all_acs,requires_grad=True).float()[:,:curr_agent.action_dim]
        if self.zero_critic:
            pol_out_params = zero_params(torch.cat((onehot_from_logits(curr_pol_out_stacked[:,:curr_agent.action_dim]),curr_pol_out_stacked[:,curr_agent.action_dim:]),dim=1))[:,curr_agent.action_dim:]
            actual_out_params = Variable(zero_params(all_acs),requires_grad=True)[:,curr_agent.action_dim:]
        else:
            pol_out_params = torch.cat((onehot_from_logits(curr_pol_out_stacked[:,:curr_agent.action_dim]),curr_pol_out_stacked[:,curr_agent.action_dim:]),dim=1)[:,curr_agent.action_dim:]
            actual_out_params = Variable(all_acs,requires_grad=True)[:,curr_agent.action_dim:]
        MSE = F.mse_loss(pol_out_params,actual_out_params)

        pol_loss = MSE + F.mse_loss(pol_out_actions,actual_out_actions)
        
        reg_param = 5.0
        entropy_reg = (-torch.log(pol_out_actions).sum(dim=1).mean() * 1e-3)/reg_param # regularize using log probabilities
        pol_loss.backward(retain_graph=True)
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_optimizer.step()

        if self.I2A:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            #labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            #self.ws_onehot.zero_() # reset OH tensor
            #self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
            agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch
            
            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            #actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew # self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        self.niter +=1
        #print(time.time() - start,"up")
        if self.niter % 100 == 0:
            self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor': np.round(pol_loss.item(),4)},
                                                                        ignore_index=True)
            print("Team (%s) Agent(%i) Policy loss" % (side, agent_i),pol_loss)

            if self.I2A:
                print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)




# ----------------------------------------------------------------------------------------

    def pretrain_actor_LSTM(self, team_sample=[], opp_sample =[], agent_i = 0, side='team', parallel=False, logger=None, act_only=False, obs_only=False,forward_pass=True,load_same_agent=False,session_path="",lstm_burn_in=40,burnin=False):
        """
        Update parameters of actor based on sample from replay buffer for policy imitation (fits policy to observed actions)
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
        if self.niter % 1000:
            self.policy_loss_logger.to_csv(session_path + 'actor_loss.csv')

        start = time.time()

        #start = time.time()
        # rews = 1-step, cum-rews = n-step
        count = self.team_count[agent_i]
        curr_agent = self.team_agents[agent_i]
        nagents = self.nagents_team
        policies = self.team_policies
        opp_policies = self.opp_policies
        obs, acs, rews, dones, MC_rews,n_step_rews,ws,rec_states,sorted_feats = team_sample 
        opp_obs, opp_acs, opp_rews, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws,_,_ = opp_sample

        self.curr_agent_index = agent_i

            
        # Zero state initialization then burn-in LSTM
        self.zero_hidden_policy(self.batch_size*nagents,self.torch_device)
        
        curr_agent.policy_optimizer.zero_grad()

        
        if self.zero_critic:
            for i in range(nagents):
                sorted_feats[i][2] = [zero_params(a) for a in sorted_feats[i][2]]
                sorted_feats[i][3] = [zero_params(a) for a in sorted_feats[i][3]]

        obs = sorted_feats[0][0]  # use features sorted by prox and stacked per agent along batch
        opp_obs =  sorted_feats[0][1]
        acs =  sorted_feats[0][2] 
        opp_acs = sorted_feats[0][3]

        if burnin:

            slice_obs = list(map(lambda x: x[:lstm_burn_in], obs))
            slice_acs = list(map(lambda x: x[:lstm_burn_in], acs))
            burn_in_obs = torch.cat(slice_obs,dim=1)
            _ = curr_agent.policy(burn_in_obs) # burn in
            # Get post-burn in training steps
            slice_obs = list(map(lambda x: x[lstm_burn_in:], obs))
            slice_acs = list(map(lambda x: x[lstm_burn_in:], acs)) 
        else:
            slice_obs = obs
            slice_acs = acs

        obs_stacked = torch.cat(slice_obs,dim=1)
        all_acs = torch.cat(slice_acs,dim=1)
        if self.I2A:
            curr_pol_out = curr_agent.policy(obs_stacked) # <--- Out of date
        else:
            curr_pol_out = curr_agent.policy(obs_stacked)
        
        
        
    
        pol_out_actions = torch.softmax(curr_pol_out[:,:,:curr_agent.action_dim],dim=2).float()
        actual_out_actions = Variable(all_acs,requires_grad=True).float()[:,:,:curr_agent.action_dim]
        
        pol_out_params = zero_params(torch.cat((onehot_from_logits(curr_pol_out[:,:,:curr_agent.action_dim],LSTM=True),curr_pol_out[:,:,curr_agent.action_dim:]),dim=2))[:,:,curr_agent.action_dim:]
        actual_out_params = Variable(zero_params(all_acs),requires_grad=True)[:,:,curr_agent.action_dim:]
        
        pol_loss = F.mse_loss(pol_out_params,actual_out_params) + F.mse_loss(pol_out_actions,actual_out_actions)
        
        reg_param = 5.0
        #entropy_reg = (-torch.log(pol_out_actions).sum(dim=2).mean() * 1e-3)/reg_param # regularize using log probabilities
        pol_loss.backward(retain_graph=False)
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_optimizer.step()

        if self.I2A:
            # Update policy prime
            curr_agent.policy_prime_optimizer.zero_grad()
            # We take the loss between the current policy's behavior and policy prime which is estimating the current policy
            pol_prime_out = [curr_agent.policy_prime(obs[ag]) for ag in range(nagents)] # uses gumbel across the actions
            pol_prime_out_stacked = torch.cat(pol_prime_out,dim=0)
            pol_prime_loss = F.mse_loss(pol_prime_out_stacked,curr_pol_out_stacked)
            pol_prime_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.policy_prime_optimizer.step()
            # Train Environment Model -----------------------------------------------------            
            curr_agent.EM_optimizer.zero_grad()
            
            #labels = ws[0].long().view(-1,1) % self.world_status_dim # categorical labels for OH
            #self.ws_onehot.zero_() # reset OH tensor
            #self.ws_onehot.scatter_(1,labels,1) # fill with OH encoding
            agents_acs = torch.cat(acs,dim=1) # cat agents actions to same row
            agents_obs = torch.cat(obs,dim=0) # stack entire batch on top of eachother for each agent
            agents_nobs = torch.cat(next_obs,dim=0)
            agents_rews = torch.cat(rews,dim=0)

            acs_repeated = agents_acs.repeat(nagents,1) # repeat actions so they may be used with each agent's observation batch
            
            EM_in = torch.cat((agents_obs,acs_repeated),dim=1)
            est_obs_diff,est_rews,est_ws = curr_agent.EM(EM_in)
            actual_obs_diff = agents_nobs - agents_obs
            actual_rews = agents_rews.view(-1,1)
            #actual_ws = self.ws_onehot.repeat(nagents,1)
            loss_obs = F.mse_loss(est_obs_diff, actual_obs_diff)
            loss_rew = F.mse_loss(est_rews, actual_rews)
            #loss_ws = CELoss(est_ws,torch.argmax(actual_ws,dim=1))
            EM_loss = self.obs_weight * loss_obs + self.rew_weight * loss_rew # self.ws_weight * loss_ws
            EM_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.policy_prime.parameters(), 1) # do we want to clip the gradients?
            curr_agent.EM_optimizer.step()
        self.niter +=1
        #print(time.time() - start,"up")
        if self.niter % 100 == 0:
            self.policy_loss_logger = self.policy_loss_logger.append({                'iteration':self.niter,
                                                                        'actor': np.round(pol_loss.item(),4)},
                                                                        ignore_index=True)
            print("Team (%s) Agent(%i) Policy loss" % (side, agent_i),pol_loss)

            if self.I2A:
                print("Team (%s) Agent(%i) Policy Prime loss" % (side, agent_i),pol_prime_loss)
                print("Team (%s) Agent(%i) Environment Model loss" % (side, agent_i),EM_loss)

 
# --------------------------------------------------------------------------

    def SIL_update(self, team_sample=[], opp_sample=[],agent_i=0,side='team', parallel=False, logger=None,load_same_agent=False):
        """
        Update parameters of agent model based on sample from replay buffer using Self-Imitation Learning update:
        sil_policy_loss = (MSE(Action,Policy(obs))) * (R - Q) if R > Q
        sil_val_loss = MSE(R,Q) if R > Q
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
            
        if side == 'team':
            count = self.team_count[agent_i]
            curr_agent = self.team_agents[agent_i]
            target_policies = self.team_target_policies
            opp_target_policies = self.opp_target_policies
            nagents = self.nagents_team
            policies = self.team_policies
            opp_policies = self.opp_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = team_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = opp_sample
        else:
            count = self.opp_count[agent_i]
            curr_agent = self.opp_agents[agent_i]
            target_policies = self.opp_target_policies
            opp_target_policies = self.team_target_policies
            nagents = self.nagents_opp
            policies = self.opp_policies
            opp_policies = self.team_policies
            obs, acs, rews, next_obs, dones,MC_rews,n_step_rews,ws = opp_sample
            opp_obs, opp_acs, opp_rews, opp_next_obs, opp_dones, opp_MC_rews, opp_n_step_rews, opp_ws = team_sample
        if load_same_agent:
            curr_agent = self.team_agents[0] 
        curr_agent.critic_optimizer.zero_grad()



        curr_pol_out = curr_agent.policy(obs[agent_i]) # uses gumbel across the actions
        #curr_pol_vf_in = torch.cat((gumbel_softmax(curr_pol_out[:,:curr_agent.action_dim], hard=True),curr_pol_out[:,curr_agent.action_dim:]),dim=1)

        obs_vf_in = torch.cat((*opp_obs,*obs),dim=1)
        acs_vf_in = torch.cat((*opp_acs,*acs),dim=1)
        vf_in = torch.cat((obs_vf_in, acs_vf_in), dim=1)


        # Train critic ------------------------
        # Uses MSE between MC values and Q if MC > Q
        if self.TD3: # 
            Q1_distr,Q2_distr = curr_agent.critic(vf_in)
            if self.D4PG:
                Q1 = curr_agent.critic.distr_to_q(Q1_distr)
                Q2 = curr_agent.critic.distr_to_q(Q2_distr)
                arg = torch.argmin(torch.stack((Q1.mean(),
                                 Q2.mean()),dim=0))
                if not arg: 
                    Q = Q1
                else:
                    Q = Q2
            else:
                Q1 = Q1_distr
                Q2 = Q2_distr
                Q = torch.min(Q1_distr,Q2_distr) # not distributional since no d4pg
        else:
            Q = curr_agent.critic(vf_in)
        # Critic assessment of current policy actions

        returns = MC_rews[agent_i].view(-1,1)

        if self.TD3: # handle double critic
            clipped_differences_Q1 = torch.clamp(returns - Q1,min=0.0).view(-1,1)
            clipped_differences_Q2 = torch.clamp(returns - Q2,min=0.0).view(-1,1)
            #print(Q)
            vf_loss = torch.norm(clipped_differences_Q1)/2.0 + torch.norm(clipped_differences_Q2)/2.0
            clipped_differences = torch.clamp(returns - Q,min=0.0).view(-1,1)
        else:
            clipped_differences = torch.clamp(returns - Q,min=0.0).view(-1,1)
            vf_loss = torch.norm(clipped_differences)
            
        vf_loss.backward(retain_graph=True) 
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 1)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.zero_grad()

        # Train actor ---------------------------

        
        
        pol_out_actions = torch.softmax(curr_pol_out[:,:curr_agent.action_dim],dim=1).float()
        actual_out_actions = Variable(torch.stack(acs)[agent_i],requires_grad=True).float()[:,:curr_agent.action_dim]
        pol_out_params = curr_pol_out[:,curr_agent.action_dim:]
        actual_out_params = Variable(torch.stack(acs)[agent_i],requires_grad=True)[:,curr_agent.action_dim:]
        #target_classes = torch.argmax(actual_out_actions,dim=1) # categorical integer for predicted class
        
        param_errors = (pol_out_params - actual_out_params)**2
        action_errors = (pol_out_actions - actual_out_actions)**2
        
        param_errors *= clipped_differences
        action_errors *= clipped_differences

        param_loss = torch.mean(param_errors)
        action_loss = torch.mean(action_errors)

        if self.D4PG:
            reg_param = 5.0
        else:
            reg_param = 1.0

        entropy_reg = (-torch.log_softmax(curr_pol_out,dim=1)[:,:curr_agent.action_dim].sum(dim=1).mean() * 1e-3)/reg_param # regularize using log probabilities

        pol_loss = action_loss + param_loss + entropy_reg # Weighted MSE based on how off Q was.
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 1) # do we want to clip the gradients?
        curr_agent.policy_optimizer.step()
        self.niter +=1
        if self.niter % 101 == 0:
            print("Team (%s) SIL Actor loss:" % side,np.round(pol_loss.item(),6))
            print("Team (%s) SIL Critic loss:" % side,np.round(vf_loss.item(),6))
        
        return clipped_differences.cpu()
        
        # ------------------------------------

       

    @classmethod
    def init_from_env(cls, env, config, agent_alg="MADDPG", adversary_alg="MADDPG", 
                        only_policy=False,reduced_obs_dim=16):
                      
        """
        Instantiate instance of this class from multi-agent environment 
        """
        team_agent_init_params = []
        team_net_params = []
        opp_agent_init_params = []
        
        opp_alg_types = [ adversary_alg for atype in range(env.num_OA)]
        team_alg_types = [ agent_alg for atype in range(env.num_TA)]
        for acsp, obsp, algtype in zip([env.action_list for i in range(env.num_TA)], env.team_obs, team_alg_types):
            
            # changed acsp to be action_list for each agent 
            # giving dimension num_TA x action_list so they may zip properly    

            if preprocess:
                num_in_pol = reduced_obs_dim
            else:
                num_in_pol = obsp.shape[0]
            num_in_reducer = obsp.shape[0]
            num_out_pol =  len(env.action_list)

            if not discrete_action:
                num_out_pol = len(env.action_list) + len(env.team_action_params[0])
            
            num_in_EM = (num_out_pol*env.num_TA) + num_in_pol
            num_out_EM = num_in_pol

            num_in_critic = (num_in_pol - num_out_pol)  + (num_out_pol * env.num_TA *2 ) + (env.num_TA -1)            
            
            team_agent_init_params.append({'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'num_in_critic': num_in_critic,
                                        'num_in_reducer': num_in_reducer})
            
            opp_agent_init_params.append({'num_in_pol': num_in_pol,
                                        'num_out_pol': num_out_pol,
                                        'num_in_critic': num_in_critic,
                                        'num_in_reducer': num_in_reducer})
            
            team_net_params.append({'num_in_pol': num_in_pol,
                                    'num_out_pol': num_out_pol,
                                    'num_in_critic': num_in_critic,
                                    'num_in_EM': num_in_EM,
                                    'num_out_EM': num_out_EM,
                                    'num_in_reducer': num_in_reducer})

        ## change for continuous
        init_dict = {'gamma': config.gamma, 'batch_size': config.batch_size,
                     'tau': config.tau, 'a_lr': config.a_lr,
                     'c_lr':configc_lr,
                     'hidden_dim': config.hidden_dim,
                     'team_alg_types': team_alg_types,
                     'opp_alg_types': opp_alg_types,
                     'device': config.device,
                     'team_agent_init_params': team_agent_init_params,
                     'opp_agent_init_params': opp_agent_init_params,
                     'team_net_params': team_net_params,
                     'discrete_action': config.discrete_action,
                     'vmax': config.vmax,
                     'vmin': config.vmin,
                     'N_ATOMS': config.n_atoms,
                     'n_steps': config.n_steps,
                     'DELTA_Z': config.delta_z,
                     'D4PG': config.d4pg,
                     'beta': config.init_beta,
                     'TD3': config.td3,
                     'TD3_noise': config.td3_noise,
                     'TD3_delay_steps': config.td3_delay,
                     'I2A': config.i2a,
                     'EM_lr': config.em_lr,
                     'obs_weight': config.obs_w,
                     'rew_weight': config.rew_w,
                     'ws_weight': config.ws_w,
                     'rollout_steps': config.roll_steps,
                     'LSTM_hidden': config.lstm_hidden,
                     'imagination_policy_branch': config.imag_pol_branch,
                     'critic_mod_both': config.crit_both,
                     'critic_mod_act': config.crit_ac,
                     'critic_mod_obs': config.crit_obs,
                     'seq_length': config.seq_length,
                     'LSTM': config.lstm_crit,
                     'LSTM_policy': config.lstm_policy,
                     'hidden_dim_lstm': config.hidden_dim_lstm,
                     'lstm_burn_in': config.burn_in_lstm,
                     'overlap': config.overlap,
                     'only_policy': only_policy,
                     'multi_gpu': config.multi_gpu,
                     'data_parallel': config.data_parallel,
                     'reduced_obs_dim':reduced_obs_dim,
                     'preprocess': config.preprocess,
                     'zero_critic': config.zero_crit,
                     'cent_critic': config.cent_crit}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    def delete_non_policy_nets(self):
        for a in self.team_agents:
            del a.target_critic
            del a.target_policy
            del a.critic
        for a in self.opp_agents:
            del a.target_critic
            del a.target_policy
            del a.critic

        
    def load_random_policy(self,side='team',nagents=1,models_path="models",load_same_agent=False):
        """
        Load new networks into the currently running session
        Returns the index chosen for each agent
        """
        folders = os.listdir(models_path)
        folders.sort()
        if not load_same_agent:
                
            ind = [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
            folder = [folder for folder in folders] # the folder names for each agents model
            filenames = []
            for i,f in zip(ind,folder):
                current = os.listdir(models_path +f)
                if side == 'team':
                    current.sort()
                filenames.append(current[i])
            
            save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
            for i in range(nagents):
                if side=='team':
                    self.team_agents[i].load_policy_params(save_dicts[i]['agent_params'])
                else:
                    self.opp_agents[i].load_policy_params(save_dicts[i]['agent_params'])
            return ind
        else:
            ind = [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
        folder = [folder for folder in folders] # the folder names for each agents model
        filenames = []
        for i,f in zip(ind,folder):
            current = os.listdir(models_path +f)
            if side == 'team':
                current.sort()
            filenames.append(current[i])
        
        save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
        for i in range(nagents):
            if side=='team':
                self.team_agents[i].load_policy_params(save_dicts[0]['agent_params'])
            else:
                self.opp_agents[i].load_policy_params(save_dicts[0]['agent_params'])
        return ind
    
    def load_random(self,side='team',nagents=1,models_path="models",load_same_agent=False):
        """
        Load new networks into the currently running session
        Returns the index chosen for each agent
        """
        folders = os.listdir(models_path)
        folders.sort()
        if not load_same_agent:
                
            ind = [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
            folder = [folder for folder in folders] # the folder names for each agents model
            filenames = []
            for i,f in zip(ind,folder):
                current = os.listdir(models_path +f)
                if side == 'team':
                    current.sort()
                filenames.append(current[i])
            
            save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
            for i in range(nagents):
                if side=='team':
                    self.team_agents[i].load_params(save_dicts[i]['agent_params'])
                else:
                    self.opp_agents[i].load_params(save_dicts[i]['agent_params'])
            return ind
        else:
            ind = [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
        folder = [folder for folder in folders] # the folder names for each agents model
        filenames = []
        for i,f in zip(ind,folder):
            current = os.listdir(models_path +f)
            if side == 'team':
                current.sort()
            filenames.append(current[i])
        
        save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
        for i in range(nagents):
            if side=='team':
                self.team_agents[i].load_params(save_dicts[0]['agent_params'])
            else:
                self.opp_agents[i].load_params(save_dicts[0]['agent_params'])
        return ind
                
                    
    def load_random_ensemble(self,side='team',nagents=1,models_path="models",load_same_agent=False):
        """
        Load new networks into the currently running session
        Returns the index chosen for each agent
        """
        folders = os.listdir(models_path)
        folders.sort()
        if not load_same_agent:
                
            ind = [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
            folder = [folder for folder in folders] # the folder names for each agents model
            filenames = []
            for i,f in zip(ind,folder):
                current = os.listdir(models_path +f)
                if side == 'team':
                    current.sort()
                filenames.append(current[i])
            
            save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
            for i in range(nagents):
                if side=='team':
                    self.team_agents[i].load_params(save_dicts[i]['agent_params'])
                else:
                    self.opp_agents[i].load_params(save_dicts[i]['agent_params'])
            return ind
        else:
            ind = [0 for _ in range(nagents)] #= [np.random.choice(np.arange(len(os.listdir(models_path + folder)))) for folder in folders] # indices of random model from each agents model folder
            folder = [folder for folder in folders] # the folder names for each agents model
            filenames = []
            for i,f in zip(ind,folder):
                current = os.listdir(models_path +f)
                if side == 'team':
                    current.sort()
                filenames.append(current[i])
            
            save_dicts = np.asarray([torch.load(models_path + fold + "/" +  filename) for filename,fold in zip(filenames,folder)]) # params for agent from randomly chosen file from model folder
            for i in range(nagents):
                if side=='team':
                    self.team_agents[i].load_params(save_dicts[i]['agent_params'])
                else:
                    self.opp_agents[i].load_params(save_dicts[i]['agent_params'])
            return [0 for _ in range(nagents)]
            
                             
    def load_ensemble(self, ensemble_path,ensemble,agentID):
        # Loads ensemble to team agent #agentID
  
        dict = torch.load(ensemble_path +("ensemble_agent_%i/model_%i.pth" % (agentID,ensemble)))
        self.team_agents[agentID].load_params(dict['agent_params'])

    def load_same_ensembles(self, ensemble_path,ensemble,nagents,load_same_agent):
        # Loads ensemble to team agent #agentID
  
        if load_same_agent:
            dicts = [torch.load(ensemble_path +("ensemble_agent_%i/model_%i.pth" % (0,ensemble))) for i in range(nagents)]
        else:
            dicts = [torch.load(ensemble_path +("ensemble_agent_%i/model_%i.pth" % (i,ensemble))) for i in range(nagents)]
        [self.team_agents[i].load_params(dicts[i]['agent_params']) for i in range(nagents)]
            
                             
    def load_ensemble_policy(self, ensemble_path,ensemble,agentID):
        # Loads ensemble to team agent #agentID
  
        dict = torch.load(ensemble_path +("ensemble_agent_%i/model_%i.pth" % (agentID,ensemble)))
        self.team_agents[agentID].load_policy_params(dict['agent_params'])
        self.team_agents[agentID].load_target_policy_params(dict['agent_params'])



                             
    def load_team(self, side='team',models_path='',load_same_agent=False,nagents=0):
        # load agent2d role
        dict = torch.load(models_path)

        if side =='team':
            [self.team_agents[i].load_params(dict['agent_params']) for i in range(nagents)]
        else:
            [self.opp_agents[i].load_params(dict['agent_params']) for i in range(nagents)]


                             
    def load_agent2d_policy(self, side='team',models_path='agent2d',load_same_agent=False,agentID=0):
        # load agent2d role
        dict = torch.load("models/" + "agent2d/agent2D.pth")

        if side =='team':
            self.team_agents[agentID].load_params(dict['agent_params'])
        else:
            self.opp_agents[agentID].load_params(dict['agent_params'])
                             
    def load_agent2d_policies(self, side='team',models_path='models/',load_same_agent=False,nagents=0):
        # load agent2d role
        dict = torch.load(models_path + "agent2d/agent2D.pth")

        if side =='team':
            [self.team_agents[i].load_params(dict['agent_params']) for i in range(nagents)]
        else:
            [self.opp_agents[i].load_params(dict['agent_params']) for i in range(nagents)]
            
    def load_agent2d(self, side='team',models_path='models/',load_same_agent=False,nagents=0):
        # load agent2d role
        dict = torch.load(models_path + "agent2d/agent2D.pth")

        if side =='team':
            [self.team_agents[i].load_params(dict['agent_params']) for i in range(nagents)]
        else:
            [self.opp_agents[i].load_params(dict['agent_params']) for i in range(nagents)]




                                          
    def first_save(self, file_path,num_copies=1):
        """
        Makes K clones of each agent to be used as the ensemble agents"""
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        [torch.save(save_dicts[i], file_path + ("ensemble_agent_%i" % i) + "/model_%i.pth" % j) for i in range(len(self.team_agents)) for j in range(num_copies)]
        self.prep_training(device=self.device,torch_device=self.torch_device)


    def save(self, filename,ep_i,torch_device=torch.device('cuda:0')):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        [torch.save(save_dicts[i], filename +("agent_%i/model_episode_%i.pth" % (i,ep_i))) for i in range(len(self.team_agents))]
        self.prep_training(device=self.device,torch_device=torch_device)
    def save_agent(self, filename,ep_i,agentID,load_same_agent,torch_device=torch.device('cuda:0')):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        if load_same_agent:
            torch.save(save_dicts[0], filename +("agent_%i/model_episode_%i.pth" % (agentID,ep_i)))

        else:
            torch.save(save_dicts[agentID], filename +("agent_%i/model_episode_%i.pth" % (agentID,ep_i)))


        self.prep_training(device=self.device,torch_device=torch_device)
        
    def save_agent2d(self, filename,agentID,load_same_agent,torch_device=torch.device('cuda:0')):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        if load_same_agent:
            torch.save(save_dicts[agentID], filename +("agent2d/agent2D.pth"))
        else:
            torch.save(save_dicts[agentID], filename +("agent2d/agent2D.pth"))

        self.prep_training(device=self.device,torch_device=torch_device)
        
    @classmethod
    def init_from_save_selfplay(cls, filenames=list,nagents=1):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        random_sessions = random.sample(filenames,nagents)
        save_dicts =  [torch.load(filename) for filename in random_sessions]
        instance = cls(**save_dicts[0]['init_dict'])
        instance.init_dict = save_dicts[0]['init_dict']
        
        for a, params in zip(instance.nagents_team, random.sample(random.sample(save_dicts,1)['agent_params'],1)):
            a.load_params(params)

        for a, params in zip(instance.opp_agents,  random.sample(random.sample(save_dicts,1)['agent_params'],1)):
            a.load_params(params)

        return instance

    @classmethod
    def init_from_save_evaluation(cls, filenames,nagents=1):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dicts = np.asarray([torch.load(filename) for filename in filenames]) # use only filename
        instance = cls(**save_dicts[0]['init_dict'])
        instance.init_dict = save_dicts[0]['init_dict']

        for i in range(nagents):
            instance.team_agents[i].load_params(save_dicts[i]['agent_params'] ) # first n agents
            

        return instance
    
    def save_ensembles(self, ensemble_path,current_ensembles):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        [torch.save(save_dicts[i], ensemble_path +("ensemble_agent_%i/model_%i.pth" % (i,j))) for i,j in zip(range(len(self.team_agents)),current_ensembles)]
        self.prep_training(device=self.device)

       
    
        #self.prep_training(device=self.device)
        
    def save_ensemble(self, ensemble_path,ensemble,agentID,load_same_agent,torch_device=torch.device('cuda:0')):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        
        save_dicts = np.asarray([{'init_dict': self.init_dict,
                     'agent_params': a.get_params() } for a in (self.team_agents)])
        if not load_same_agent:
            torch.save(save_dicts[agentID], ensemble_path +("ensemble_agent_%i/model_%i.pth" % (agentID,ensemble)))
        else:
            torch.save(save_dicts[0], ensemble_path +("ensemble_agent_%i/model_%i.pth" % (agentID,ensemble)))
            
       
        self.prep_training(device=self.device,torch_device=torch_device)
    

        

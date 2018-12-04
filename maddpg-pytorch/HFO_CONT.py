import re
import itertools
import random
import datetime
import os 
import csv
import itertools 
import argparse
#import tensorflow.contrib.slim as slim
import numpy as np
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy,zero_params,pretrain_process
from torch import Tensor
from HFO import hfo
import time
import _thread as thread
import torch
from pathlib import Path
from torch.autograd import Variable#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from HFO_env import *
from utils.trainer import launch_eval
parser = argparse.ArgumentParser(description='Load port and log directory')
parser.add_argument('-port', type=int,default=6000,
                   help='An integer for port number')
parser.add_argument('-log_dir', type=str, default='log',
                   help='A name for log directory')
parser.add_argument('-log', type=str, default='history',
                   help='A name for log file ')

args = parser.parse_args()
log_dir = args.log_dir
port = args.port
history = args.log

    
# options ------------------------------
action_level = 'low'
feature_level = 'low'
USE_CUDA = True 
if USE_CUDA:
    device = 'cuda'
    to_gpu = True
else:
    to_gpu = False
    device = 'cpu'

use_viewer = True
n_training_threads = 8
<<<<<<< HEAD
use_viewer_after = 10 # If using viewer, uses after x episodes
=======
use_viewer_after = 5 # If using viewer, uses after x episodes
>>>>>>> 0e38562b7759b3e6a230ac695aab64c817389fa6
# default settings
num_episodes = 100000
replay_memory_size = 1000000
episode_length = 500 # FPS
untouched_time = 500
burn_in_iterations = 500 # for time step
burn_in_episodes = float(burn_in_iterations)/episode_length
# --------------------------------------
# hyperparams--------------------------
<<<<<<< HEAD
batch_size = 1
=======
batch_size = 128
>>>>>>> 0e38562b7759b3e6a230ac695aab64c817389fa6
hidden_dim = int(1024)
a_lr = 0.00005 # actor learning rate
c_lr = 0.0005 # critic learning rate
tau = 0.005 # soft update rate
steps_per_update = 2
# exploration --------------------------
explore = True
final_OU_noise_scale = 0.1
final_noise_scale = 0.1
init_noise_scale = 1.00
num_explore_episodes = 25  # Haus uses over 10,000 updates --
# --------------------------------------
#D4PG Options --------------------------
D4PG = True
gamma = 0.99 # discount
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
n_steps = 5 # n-step update size 
# Mixed taqrget beta (0 = 1-step, 1 = MC update)
initial_beta = 1.0
final_beta = 0.0 #
num_beta_episodes = 1000
#---------------------------------------
train_team = True
train_opp = False
#---------------------------------------
#TD3 Options ---------------------------
TD3 = True
TD3_delay_steps = 5
TD3_noise = 0.01
# --------------------------------------
#Pretrain Options ----------------------
# To use imitation exporation run 1 TNPC vs 0/1 ONPC (currently set up for 1v1, or 1v0)
# Copy the base_left-11.log to Pretrain_Files and rerun this file with 1v1 or 1v0 controlled vs npc respectively
Imitation_exploration = False
test_imitation = False  # After pretrain, infinitely runs the current pretrained policy
pt_critic_updates = 5000
pt_actor_updates = 5000
pt_actor_critic_updates = 0
pt_imagination_branch_pol_updates = 5000
pt_episodes = 3000# num of episodes that you observed in the gameplay between npcs
pt_EM_updates = 30000
pt_beta = 1.0
#---------------------------------------
#I2A Options ---------------------------
I2A = False
decent_EM = True
EM_lr = 0.005
decent_EM = True
obs_weight = 10.0
rew_weight = 1.0
ws_weight = 1.0
rollout_steps = 1
LSTM_hidden=16
imagination_policy_branch = False
#---------------------------------------
# Self-Imitation Learning Options ------
SIL = False
SIL_update_ratio = 3
#---------------------------------------
# Self-play ----------------------------
#Save/load -----------------------------
save_nns = True
ep_save_every = 2 # episodes
#Load previous NNs, currently set to load only at initialization.
load_nets = False
load_path = "models"
folder_path = None
current_day_time = datetime.datetime.now()
first_save = False
# --------------------------------------
# Evaluation ---------------------------
evaluate = True
eval_after = 2
eval_episodes = 5
#---------------------------------------
# initialization -----------------------
t = 0
time_step = 0
kickable_counter = 0
# if using low level actions use non discrete settings
if action_level == 'high':
    discrete_action = True
else:
    discrete_action = False
if not USE_CUDA:
        torch.set_num_threads(n_training_threads)
    
env = HFO_env(num_TNPC = 0,num_TA=2,num_OA=2, num_ONPC=0, num_trials = num_episodes, fpt = episode_length, # create environment
              feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,offense_on_ball=False,port=port,log_dir=log_dir)

if use_viewer:
    env._start_viewer()

if save_nns:
    folder_path = 'models/' + str(env.num_TA) + '_vs_' + str(env.num_OA) + '/time' + \
                                '_' + str(current_day_time.month) + \
                                '_' + str(current_day_time.day) + \
                                '_' + str(current_day_time.hour) +'/'
if not os.path.exists(os.path.dirname(folder_path)):
    try:
        os.makedirs(os.path.dirname(folder_path))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

time.sleep(3)
print("Done connecting to the server ")

# initializing the maddpg 
if load_nets:
    maddpg = MADDPG.init_from_save(load_path,num_TA)
else:
    maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                              adversary_alg= "MADDPG",device=device,
                              gamma=gamma,batch_size=batch_size,
                              tau=tau,
                              a_lr=a_lr,
                              c_lr=c_lr,
                              hidden_dim=hidden_dim ,discrete_action=discrete_action,
                              vmax=Vmax,vmin=Vmin,N_ATOMS=N_ATOMS,
                              n_steps=n_steps,DELTA_Z=DELTA_Z,D4PG=D4PG,beta=initial_beta,
                              TD3=TD3,TD3_noise=TD3_noise,TD3_delay_steps=TD3_delay_steps,
                              I2A = I2A, EM_lr = EM_lr,
                              obs_weight = obs_weight, rew_weight = rew_weight, ws_weight = ws_weight, 
                              rollout_steps = rollout_steps,LSTM_hidden=LSTM_hidden,decent_EM = decent_EM,imagination_policy_branch = imagination_policy_branch)


# print('maddpg.nagents ', maddpg.nagents)
# print('env.num_TA ', env.num_TA)  
# print('env.num_features : ' , env.num_features)
#initialize the replay buffer of size 10000 for number of agent with their observations & actions 

#pretrain_buffer = ReplayBuffer(replay_memory_size , env.num_TA,
#                                 [env.team_num_features for i in range(env.num_TA)],
#                                 [env.team_action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)])

team_replay_buffer = ReplayBuffer(replay_memory_size , env.num_TA,
                                 [env.team_num_features for i in range(env.num_TA)],
                                 [env.team_action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)])

#initialize the replay buffer of size 10000 for number of opponent agent with their observations & actions 
opp_replay_buffer = ReplayBuffer(replay_memory_size , env.num_OA,
                                 [env.opp_num_features for i in range(env.num_OA)],
                                 [env.opp_action_params.shape[1] + len(env.action_list) for i in range(env.num_OA)])

reward_total = [ ]
num_steps_per_episode = []
end_actions = [] 
# logger_df = pd.DataFrame()
team_step_logger_df = pd.DataFrame()
opp_step_logger_df = pd.DataFrame()


# -------------------------------------
# PRETRAIN ############################
if Imitation_exploration:

    team_pt_obs = []
    team_pt_statuses = []
    team_pt_actions = []

    opp_pt_obs = []
    opp_pt_statuses = []
    opp_pt_actions = []
    
    pt_ob, pt_status,pt_action = pretrain_process(fname = 'Pretrain_Files/base_left-11.log',pt_episodes = pt_episodes,episode_length = episode_length,num_features = env.num_features)
    team_pt_obs.append(pt_ob)
    team_pt_statuses.append(pt_status)
    team_pt_actions.append(pt_action)
    
    pt_ob, pt_status,pt_action = pretrain_process(fname = 'Pretrain_Files/base_left-7.log',pt_episodes = pt_episodes,episode_length = episode_length,num_features = env.num_features)
    team_pt_obs.append(pt_ob)
    team_pt_statuses.append(pt_status)
    team_pt_actions.append(pt_action)
    
        
    pt_ob, pt_status,pt_action = pretrain_process(fname = 'Pretrain_Files/base_right-1.log',pt_episodes = pt_episodes,episode_length = episode_length,num_features = env.num_features)
    opp_pt_obs.append(pt_ob)
    opp_pt_statuses.append(pt_status)
    opp_pt_actions.append(pt_action)
    
    pt_ob, pt_status,pt_action = pretrain_process(fname = 'Pretrain_Files/base_right-2.log',pt_episodes = pt_episodes,episode_length = episode_length,num_features = env.num_features)
    opp_pt_obs.append(pt_ob)
    opp_pt_statuses.append(pt_status)
    opp_pt_actions.append(pt_action)
    
    print("Length of obs,stats,actions",len(pt_obs),len(pt_status),len(pt_actions))
    time_step = 0
    for ep_i in range(0, pt_episodes):
        if ep_i % 100 == 0:
            print("Pushing Pretrain Episode:",ep_i)
            
            
            
        # team n-step
        team_n_step_rewards = []
        team_n_step_obs = []
        team_n_step_acs = []
        team_n_step_next_obs = []
        team_n_step_dones = []
        team_n_step_ws = []


        # opp n-step
        opp_n_step_rewards = []
        opp_n_step_obs = []
        opp_n_step_acs = []
        opp_n_step_next_obs = []
        opp_n_step_dones = []
        opp_n_step_ws = []

        n_step_rewards = []
        n_step_reward = 0.0
        n_step_obs = []
        n_step_acs = []
        n_step_next_obs = []
        n_step_dones = []
        n_step_ws = []
        maddpg.prep_rollouts(device=device)
        #define/update the noise used for exploration
        explr_pct_remaining = 0.0
        beta_pct_remaining = 0.0
        maddpg.scale_q(0.0)
        maddpg.reset_noise()
        maddpg.scale_beta(pt_beta)
        d = False
        for et_i in range(0, episode_length):
            agent_actions = [pt_actions[time_step]]
            obs =  np.array([pt_obs[time_step] for i in range(maddpg.nagents)]).T
            world_stat = pt_status[time_step]
            d = False
            if world_stat != 0.0:
                d = True
            next_obs =  np.array([pt_obs[time_step+1] for i in range(maddpg.nagents)]).T

            rewards = np.hstack([env.getPretrainRew(world_stat,i,d,obs,next_obs) for i in range(env.num_TA) ])            
            dones = np.hstack([d for i in range(env.num_TA)])

            # Store variables for calculation of MC and n-step targets
            n_step_rewards.append(rewards)
            n_step_obs.append(obs)
            n_step_next_obs.append(next_obs)
            n_step_acs.append(agent_actions)
            n_step_dones.append(dones)
            time_step += 1
            if d == True: # Episode done
                # Calculate n-step and MC targets
                for n in range(et_i+1):
                    MC_target = 0
                    n_step_target = 0
                    n_step_ob = n_step_obs[n]
                    n_step_ac = n_step_acs[n]    
                    for step in range(et_i+1 - n): # sum MC target
                        MC_target += n_step_rewards[et_i - step] * gamma**(et_i - n - step)
                    if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                        for step in range(n_steps): 
                            n_step_target += n_step_rewards[n + step] * gamma**(step)
                        n_step_next_ob = n_step_next_obs[n - 1 + n_steps]
                        n_step_done = n_step_dones[n - 1 + n_steps]
                    else: # n-step = MC if less than n steps remaining
                        n_step_target = MC_target
                        n_step_next_ob = n_step_next_obs[-1]
                        n_step_done = n_step_dones[-1]
                    # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                    #pretrain_buffer.push
                    replay_buffer.push(n_step_ob, n_step_ac,n_step_rewards[n],n_step_next_ob,n_step_done,
                                         np.hstack([MC_target]),np.hstack([n_step_target]),np.hstack([world_stat])) 
                time_step +=1
                break

                maddpg.prep_training(device=device)
    
    if I2A:
        # pretrain EM
        for i in range(pt_EM_updates):
            if i%100 == 0:
                print("Petrain EM update:",i)
            for u_i in range(1):
                for a_i in range(maddpg.nagents):
                    #sample = pretrain_buffer.sample(batch_size,
                    sample = replay_buffer.sample(batch_size,
                                                  to_gpu=to_gpu,norm_rews=False)
                    maddpg.update_EM(sample, a_i,'team' )
                    maddpg.update_EM(sample, a_i,'opp' )


    if I2A:
        if Imitation_exploration:
            # pretrain policy prime
            for i in range(pt_actor_updates):
                if i%100 == 0:
                    print("Petrain prime update:",i)
                for u_i in range(1):
                    for a_i in range(maddpg.nagents):
                        #sample = pretrain_buffer.sample(batch_size,
                        sample = replay_buffer.sample(batch_size,
                                                      to_gpu=to_gpu,norm_rews=False)
                        maddpg.pretrain_prime(sample, a_i,'team')
                        maddpg.pretrain_prime(sample, a_i,'opp')

                     
    # pretrain policy
    for i in range(pt_actor_updates):
        if i%100 == 0:
            print("Petrain actor update:",i)
        for u_i in range(1):
            for a_i in range(maddpg.nagents):
                #sample = pretrain_buffer.sample(batch_size,
                sample = replay_buffer.sample(batch_size,
                                              to_gpu=to_gpu,norm_rews=False)
                maddpg.pretrain_actor(sample, a_i,'team')
                maddpg.pretrain_actor(sample, a_i,'opp')

    maddpg.update_hard_policy()
    
    
    # pretrain critic
    for i in range(pt_critic_updates):
        if i%100 == 0:
            print("Petrain critic update:",i)
        for u_i in range(1):
            for a_i in range(maddpg.nagents):
                #sample = pretrain_buffer.sample(batch_size,
                sample = replay_buffer.sample(batch_size,
                                                to_gpu=to_gpu,norm_rews=False)
                maddpg.pretrain_critic(sample, a_i,'team')
                maddpg.pretrain_critic(sample, a_i,'opp')
    maddpg.update_hard_critic()

    # pretrain true actor-critic (non-imitation) + policy prime
    for i in range(pt_actor_critic_updates):
        if i%100 == 0:
            print("Petrain critic/actor update:",i)
        for u_i in range(1):
            for a_i in range(maddpg.nagents):
                #sample = pretrain_buffer.sample(batch_size,
                sample = replay_buffer.sample(batch_size,
                                              to_gpu=to_gpu,norm_rews=False)
                maddpg.update(sample, a_i, 'team' )
                maddpg.update(sample, a_i, 'opp' )
                maddpg.update_all_targets()
                if SIL:
                    [maddpg.SIL_update(sample, a_i,'team') for i in range(SIL_update_ratio)]
                    [maddpg.SIL_update(sample, a_i,'opp') for i in range(SIL_update_ratio)]
            maddpg.update_all_targets()
    maddpg.update_hard_critic()
    maddpg.update_hard_policy()
    
    if use_viewer:
        env._start_viewer()       

# END PRETRAIN ###################
# --------------------------------
env.launch()
time.sleep(2)
for ep_i in range(0, num_episodes):

    # team n-step
    team_n_step_rewards = []
    team_n_step_obs = []
    team_n_step_acs = []
    team_n_step_next_obs = []
    team_n_step_dones = []
    team_n_step_ws = []


    # opp n-step
    opp_n_step_rewards = []
    opp_n_step_obs = []
    opp_n_step_acs = []
    opp_n_step_next_obs = []
    opp_n_step_dones = []
    opp_n_step_ws = []
    #maddpg.prep_rollouts(device='cpu')
              

    
    
    #define/update the noise used for exploration
    if ep_i < burn_in_episodes:
        explr_pct_remaining = 1.0
    else:
        explr_pct_remaining = max(0, num_explore_episodes - ep_i + burn_in_episodes) / (num_explore_episodes)
    beta_pct_remaining = max(0, num_beta_episodes - ep_i + burn_in_episodes) / (num_beta_episodes)
    
    # evaluation for 10 episodes every 100
    if ep_i % 10 == 0:
        maddpg.scale_noise(final_OU_noise_scale + (init_noise_scale - final_OU_noise_scale) * explr_pct_remaining)
    if ep_i % 100 == 0:
        maddpg.scale_noise(0.0)

        
    maddpg.reset_noise()
    maddpg.scale_beta(final_beta + (initial_beta - final_beta) * beta_pct_remaining)
    #for the duration of 100 episode with maximum length of 500 time steps
    time_step = 0
    team_kickable_counter = 0
    opp_kickable_counter = 0
    for et_i in range(0, episode_length):

        maddpg.prep_training(device=device) # GPU for forward passes?? 

        # gather all the observations into a torch tensor 
        torch_obs_team = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                                requires_grad=False)
                        for i in range(maddpg.nagents_team)]

        # gather all the opponent observations into a torch tensor 
        torch_obs_opp = [Variable(torch.Tensor(np.vstack(env.Observation(i,'opp')).T),
                                requires_grad=False)
                        for i in range(maddpg.nagents_opp)]

        # get actions as torch Variables for both team and opp
        team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_opp, explore=explore)
        # convert actions to numpy arrays
        team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
        #Converting actions to numpy arrays for opp agents
        opp_agent_actions = [ac.cpu().data.numpy() for ac in opp_torch_agent_actions]

        # rearrange actions to be per environment
        team_params = np.asarray([ac[0][len(env.action_list):] for ac in team_agent_actions]) 
        # rearrange actions to be per environment for the opponent
        opp_params = np.asarray([ac[0][len(env.action_list):] for ac in opp_agent_actions])

        # this is returning one-hot-encoded action for each team agent
        team_actions = [[ac[0][:len(env.action_list)] for ac in team_agent_actions]]
        # this is returning one-hot-encoded action for each opp agent 
        opp_actions = [[ac[0][:len(env.action_list)] for ac in opp_agent_actions]]
        
        tensors = []
        rands = []
        if explore:
            team_noisey_actions = [e_greedy(torch.tensor(a).view(env.num_TA,len(env.action_list)), env.num_TA,
                                                 eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)) for a in team_actions]

            opp_noisey_actions = [e_greedy(torch.tensor(a).view(env.num_OA,len(env.action_list)), env.num_OA,
                                                 eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)) for a in opp_actions]
        else:
            team_noisey_actions = [e_greedy(torch.tensor(a).view(env.num_TA,len(env.action_list)), env.num_TA, eps = 0) for a in team_actions]
            opp_noisey_actions = [e_greedy(torch.tensor(a).view(env.num_OA,len(env.action_list)), env.num_OA, eps = 0) for a in opp_actions]

        team_randoms = [team_noisey_actions[0][1][i] for i in range(env.num_TA)]
        # team_noisey_actions = [team_noisey_actions[0][0][i] for i in range(env.num_TA)]

        opp_randoms = [opp_noisey_actions[0][1][i] for i in range(env.num_OA)]
        # opp_noisey_actions = [opp_noisey_actions[0][0][i] for i in range(env.num_OA)]
        
        # ***********************May use in future**************************************
        # team_noisey_actions_for_buffer = [ac.data.numpy() for ac in team_noisey_actions]
        # team_noisey_actions_for_buffer = np.asarray([ac[0] for ac in team_noisey_actions_for_buffer])

        # opp_noisey_actions_for_buffer = [ac.data.numpy() for ac in opp_noisey_actions]
        # opp_noisey_actions_for_buffer = np.asarray([ac[0] for ac in opp_noisey_actions_for_buffer])

        team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
        opp_obs =  np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T
        
        # use random unif parameters if e_greedy
        team_noisey_actions_for_buffer = np.asarray([[val for val in (np.random.uniform(-1,1,3))] if ran else action for ran,action in zip(team_randoms,team_actions[0])])
        team_params = np.asarray([[val for val in (np.random.uniform(-1,1,5))] if ran else p for ran,p in zip(team_randoms,team_params)])
        
        opp_noisey_actions_for_buffer = np.asarray([[val for val in (np.random.uniform(-1,1,3))] if ran else action for ran,action in zip(opp_randoms,opp_actions[0])])
        opp_params = np.asarray([[val for val in (np.random.uniform(-1,1,5))] if ran else p for ran,p in zip(opp_randoms,opp_params)])

        # print('These are the opp params', str(opp_params))

        team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes
        team_actions_params_for_buffer = np.array([[np.concatenate((ac,pm),axis=0) for ac,pm in zip(team_noisey_actions_for_buffer,team_params)] for i in range(1)]).reshape(
            env.num_TA,env.team_action_params.shape[1] + len(env.action_list)) # concatenated actions, params for buffer
        
        opp_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in opp_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes
        opp_actions_params_for_buffer = np.array([[np.concatenate((ac,pm),axis=0) for ac,pm in zip(opp_noisey_actions_for_buffer,opp_params)] for i in range(1)]).reshape(
            env.num_OA,env.opp_action_params.shape[1] + len(env.action_list)) # concatenated actions, params for buffer

        # If kickable is True one of the teammate agents has possession of the ball
        kickable = False
        kickable = np.array([env.get_kickable_status(i,team_obs.T) for i in range(env.num_TA)]).any()
        if kickable == True:
            team_kickable_counter += 1
            
        # If kickable is True one of the teammate agents has possession of the ball
        kickable = False
        kickable = np.array([env.get_kickable_status(i,opp_obs.T) for i in range(env.num_OA)]).any()
        if kickable == True:
            opp_kickable_counter += 1


        _,_,_,_,d,world_stat = env.Step(team_agents_actions, opp_agents_actions, team_params, opp_params)

        team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])
        opp_rewards = np.hstack([env.Reward(i,'opp') for i in range(env.num_OA)])

        team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
        opp_next_obs = np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T

        
        team_done = env.d
        opp_done = env.d 

        # Store variables for calculation of MC and n-step targets for team
        team_n_step_rewards.append(team_rewards)
        team_n_step_obs.append(team_obs)
        team_n_step_next_obs.append(team_next_obs)
        team_n_step_acs.append(team_actions_params_for_buffer)
        team_n_step_dones.append(team_done)
        team_n_step_ws.append(world_stat)
        
        opp_n_step_rewards.append(opp_rewards)
        opp_n_step_obs.append(opp_obs)
        opp_n_step_next_obs.append(opp_next_obs)
        opp_n_step_acs.append(opp_actions_params_for_buffer)
        opp_n_step_dones.append(opp_done)
        opp_n_step_ws.append(world_stat)
        # ----------------------------------------------------------------


        # print('This is the length of the buffers', str(len(team_replay_buffer)), str(len(opp_replay_buffer)))
        training = ((len(team_replay_buffer) >= batch_size and len(opp_replay_buffer) >= batch_size and 
                    (t % steps_per_update) < 1) and t > burn_in_iterations)

        if training:
            #print('****************We are now training*********************')
            maddpg.prep_training(device=device)
            for u_i in range(1):
                if train_team: # train team
                    for a_i in range(maddpg.nagents_team):
                        sample = team_replay_buffer.sample(batch_size,
                                                      to_gpu=to_gpu,norm_rews=False)
                        maddpg.update(sample, a_i, 'team')
                        if SIL:
                            [maddpg.SIL_update(sample, a_i,'team') for i in range(SIL_update_ratio)]

                if train_opp: # train opp team
                    for a_i in range(maddpg.nagents_opp):
                        sample = opp_replay_buffer.sample(batch_size,
                                                      to_gpu=to_gpu,norm_rews=False)
                        maddpg.update(sample, a_i, 'opp')
                        if SIL:
                            [maddpg.SIL_update(sample, a_i,'opp') for i in range(SIL_update_ratio)]
                maddpg.update_all_targets()
            # maddpg.prep_rollouts(device='cpu') convert back to cpu for pushing?
            
                     
        time_step += 1
        t += 1
        if t%1000 == 0:
                      
            team_step_logger_df.to_csv('team_%s.csv' % history)
            opp_step_logger_df.to_csv('opp_%s.csv' % history)
                      
                    
        if d == True: # Episode done
            # Calculate n-step and MC targets
            # ws = [world_stat] * env.num_TA
            for n in range(et_i+1):
                MC_targets = [] # gather for each agent, and push to buffer once with all agents
                n_step_targets = []
                for a in range(env.num_TA):
                    MC_target = 0
                    n_step_target = 0
                    
                    for step in range(et_i+1 - n): # sum MC target
                        MC_target += team_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
                    MC_targets.append(MC_target)
                    if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                        for step in range(n_steps): 
                            n_step_target += team_n_step_rewards[n + step][a] * gamma**(step)
                        n_step_targets.append(n_step_target)
                        n_step_next_ob = team_n_step_next_obs[n - 1 + n_steps]
                        n_step_done = team_n_step_dones[n - 1 + n_steps]
                    else: # n-step = MC if less than n steps remaining
                        n_step_target = MC_target
                        n_step_targets.append(n_step_target)
                        n_step_next_ob = team_n_step_next_obs[-1]
                        n_step_done = team_n_step_dones[-1] 
                    # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],
                                        n_step_next_ob,[n_step_done for i in range(env.num_TA)],MC_targets,
                                        n_step_targets,[team_n_step_ws[n] for i in range(env.num_TA)])
                                        

            # ws = [world_stat] * env.num_OA
            for n in range(et_i+1):
                MC_targets = []
                n_step_targets = []
                for a in range(env.num_OA):
                    MC_target = 0
                    n_step_target = 0
                    for step in range(et_i+1 - n): # sum MC target
                        MC_target += opp_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
                    MC_targets.append(MC_target)
                    if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                        for step in range(n_steps): 
                            n_step_target += opp_n_step_rewards[n + step][a] * gamma**(step)
                        n_step_targets.append(n_step_target)
                        n_step_next_ob = opp_n_step_next_obs[n - 1 + n_steps]
                        n_step_done = opp_n_step_dones[n - 1 + n_steps]
                    else: # n-step = MC if less than n steps remaining
                        n_step_target = MC_target
                        n_step_targets.append(n_step_target)
                        n_step_next_ob = opp_n_step_next_obs[-1]
                        n_step_done = opp_n_step_dones[-1]
                    # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                
                opp_replay_buffer.push(opp_n_step_obs[n], opp_n_step_acs[n],opp_n_step_rewards[n],
                                        n_step_next_ob,[n_step_done for i in range(env.num_OA)],MC_targets,
                                        n_step_targets,[opp_n_step_ws[n] for i in range(env.num_OA)])
                                       
                    
            # log
            if time_step > 0 and ep_i > 1:
                team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                        'why': env.team_envs[0].statusToString(world_stat),
                                                        'kickable_percentages': (team_kickable_counter/time_step) * 100,
                                                        'average_reward': team_replay_buffer.get_average_rewards(time_step),
                                                        'cumulative_reward': team_replay_buffer.get_cumulative_rewards(time_step),
                                                        'goals_scored': env.scored_counter_left/env.num_TA}, 
                                                        ignore_index=True)

                opp_step_logger_df = opp_step_logger_df.append({'time_steps': time_step, 
                                                        'why': env.opp_team_envs[0].statusToString(world_stat),
                                                        'kickable_percentages': (opp_kickable_counter/time_step) * 100,
                                                        'average_reward': opp_replay_buffer.get_average_rewards(time_step),
                                                        'cumulative_reward': opp_replay_buffer.get_cumulative_rewards(time_step),
                                                        'goals_scored': env.scored_counter_right/env.num_OA}, 
                                                        ignore_index=True)
                
  
            if first_save:
                if ep_i > 1 and ep_i%ep_save_every == 0 and save_nns:

                    file_name = folder_path + 'current_model_' + str(ep_i)
                    maddpg.first_save(file_name,num_copies = 3)
                first_save = False
            else:
                if ep_i > 1 and ep_i%ep_save_every == 0 and save_nns:
                    file_name = folder_path + 'model_episode_' + str(ep_i) 
                    maddpg.save(file_name)

#            if ep_i > 1 and ep_i % eval_after == 0 and evaluate:
#                
#                cmd = "python trainer.py -port2 %i -log_dir2 %s -log2 %s -eval_episodes %i"\
#                  " -ONPC %i -num_TA %i -device2 %s -filename2 %s" \
#                  % (("evaluation_ep" + str(ep_i)),"evaluation_log_dir",7000,eval_episodes,env.ONPC,
#                     env.num_TA,device,(file_name + "_agent_"))
#                self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
#                time.sleep(2) # Wait for server to startup before connecting a player

                
#                launch_eval(filenames=[file_name + ("_agent_%i" % i) + ".pth" for i in range(env.num_TA)],eval_episodes = eval_episodes,log_dir = "",log = "evaluation_ep" + str(ep_i),port = 8000,
#                            num_TA = env.num_TA, num_ONPC = env.num_OA,device=device)
            break;  
        team_obs = team_next_obs
        opp_obs = opp_next_obs

            #print(step_logger_df) 
        #if t%30000 == 0 and use_viewer:
        if t%20000 == 0 and use_viewer and ep_i > use_viewer_after:
            env._start_viewer()       

      
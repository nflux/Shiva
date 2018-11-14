import re
import itertools
import random
import datetime
import os 
import csv
import itertools 
#import tensorflow.contrib.slim as slim
import numpy as np
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy,zero_params,pretrain_process
from torch import Tensor
import hfo
import time
import _thread as thread
import argparse
import torch
from pathlib import Path
from torch.autograd import Variable#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from HFO_env import *

# options ------------------------------
action_level = 'low'
feature_level = 'low'
USE_CUDA = False 
use_viewer = True
n_training_threads = 8
use_viewer_after = 1500 # If using viewer, uses after x episodes
# default settings
num_episodes = 100000
replay_memory_size = 1000000
episode_length = 500 # FPS
untouched_time = 500
burn_in_iterations = 500 # for time step
burn_in_episodes = float(burn_in_iterations)/episode_length
# --------------------------------------
# hyperparams---------------------------
batch_size = 256
hidden_dim = int(1024)
a_lr = 0.00005 # actor learning rate
c_lr = 0.001 # critic learning rate
tau = 0.005 # soft update rate
steps_per_update = 2
# exploration --------------------------
explore = True
final_OU_noise_scale = 0.1
final_noise_scale = 0.1
init_noise_scale = 1.00
num_explore_episodes = 5  # Haus uses over 10,000 updates --
# --------------------------------------
#D4PG Options --------------------------
D4PG = True
gamma = 0.99 # discount
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
n_steps = 5 # n-step update size
# Mixed target beta (0 = 1-step, 1 = MC update)
initial_beta = 0.2
final_beta = 0.0 #
num_beta_episodes = 20
#---------------------------------------
#TD3 Options ---------------------------
TD3 = True
TD3_delay_steps = 5
TD3_noise = 0.05
# --------------------------------------
#Pretrain Options ----------------------
# To use imitation exporation run 1 TNPC vs 0/1 ONPC (currently set up for 1v1, or 1v0)
# Copy the base_left-11.log to Pretrain_Files and rerun this file with 1v1 or 1v0 controlled vs npc respectively
Imitation_exploration = True
pt_critic_updates = 10
pt_actor_updates = 25000
pt_episodes = 10 # num of episodes that you observed in the gameplay between npcs
pt_beta = 0.5
#---------------------------------------
#Save/load -----------------------------
save_critic = False
save_actor = False
#The NNs saved every #th episode.
ep_save_every = 20
#Load previous NNs, currently set to load only at initialization.
load_critic = False
load_actor = False
# --------------------------------------
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
    
env = HFO_env(num_TNPC = 0,num_TA=1, num_ONPC=1, num_trials = num_episodes, fpt = episode_length, 
              feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,offense_on_ball=False)

if use_viewer:
    env._start_viewer()

time.sleep(3)
print("Done connecting to the server ")

# initializing the maddpg 
maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=tau,
                                  a_lr=a_lr,
                                  c_lr=c_lr,
                                  hidden_dim=hidden_dim ,discrete_action=discrete_action,
                                  vmax=Vmax,vmin=Vmin,N_ATOMS=N_ATOMS,
                              n_steps=n_steps,DELTA_Z=DELTA_Z,D4PG=D4PG,beta=initial_beta,
                              TD3=TD3,TD3_noise=TD3_noise,TD3_delay_steps=TD3_delay_steps)



print('maddpg.nagents ', maddpg.nagents)
print('env.num_TA ', env.num_TA)  
print('env.num_features : ' , env.num_features)
#initialize the replay buffer of size 10000 for number of agent with their observations & actions 
pretrain_buffer = ReplayBuffer(replay_memory_size , env.num_TA,
                                 [env.num_features for i in range(env.num_TA)],
                                 [env.action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)])
replay_buffer = ReplayBuffer(replay_memory_size , env.num_TA,
                                 [env.num_features for i in range(env.num_TA)],
                                 [env.action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)])

reward_total = [ ]
num_steps_per_episode = []
end_actions = []
logger_df = pd.DataFrame()
step_logger_df = pd.DataFrame()
# -------------------------------------
# PRETRAIN ############################
if Imitation_exploration:

    pt_obs, pt_status,pt_actions = pretrain_process(fname = 'Pretrain_Files/base_left-11.log',pt_episodes = pt_episodes,episode_length = episode_length,num_features = env.num_features)
    time_step = 1
    for ep_i in range(0, pt_episodes):
        if ep_i % 100 == 0:
            print("Pushing Pretrain Episode:",ep_i)
        n_step_rewards = []
        n_step_reward = 0.0
        n_step_obs = []
        n_step_acs = []
        n_step_next_obs = []
        n_step_dones = []
        maddpg.prep_rollouts(device='cpu')
        #define/update the noise used for exploration
        explr_pct_remaining = 0.0
        beta_pct_remaining = 0.0
        maddpg.scale_noise(0.0)
        maddpg.reset_noise()
        maddpg.scale_beta(pt_beta)
        d = False
        for et_i in range(0, episode_length):
            agent_actions = [pt_actions[time_step]]
            obs =  np.array([pt_obs[time_step] for i in range(maddpg.nagents)]).T
            next_obs =  np.array([pt_obs[time_step+1] for i in range(maddpg.nagents)]).T
            world_stat = pt_status[time_step-1]
            d = False
            if world_stat != 0.0:
                d = True
            rewards = np.hstack([env.getPretrainRew(world_stat,i,d) for i in range(env.num_TA) ])            
            dones = np.hstack([d for i in range(env.num_TA)])

            print("R",rewards)
            print("D",dones)
            print("A",agent_actions)
            print("O",obs)
            print("NO",next_obs)
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
                    pretrain_buffer.push(n_step_ob, n_step_ac,n_step_rewards[n],n_step_next_ob,n_step_done,np.hstack([MC_target]),np.hstack([n_step_target])) 
                break

    # update critic and policy
    for i in range(pt_critic_updates):
        if i%100 == 0:
            print("Petrain critic/actor update:",i)
        for u_i in range(1):
            for a_i in range(maddpg.nagents):
                sample = pretrain_buffer.sample(batch_size,
                                              to_gpu=False,norm_rews=False)
                maddpg.update_critic(sample, a_i )
                maddpg.update_actor(sample, a_i )
            maddpg.update_all_targets()
        maddpg.prep_rollouts(device='cpu')
    maddpg.update_hard_critic()
    maddpg.update_hard_policy()

    if use_viewer:
        env._start_viewer()       

# END PRETRAIN ###################
# --------------------------------
env.launch()
# for the duration of 1000 episodes 
for ep_i in range(0, num_episodes):

    n_step_rewards = []
    n_step_reward = 0.0
    n_step_obs = []
    n_step_acs = []
    n_step_next_obs = []
    n_step_dones = []
    maddpg.prep_rollouts(device='cpu')
    #define/update the noise used for exploration
    if ep_i < burn_in_episodes:
        explr_pct_remaining = 1.0
    else:
        explr_pct_remaining = max(0, num_explore_episodes - ep_i + burn_in_episodes) / (num_explore_episodes)
    beta_pct_remaining = max(0, num_beta_episodes - ep_i + burn_in_episodes) / (num_beta_episodes)
    maddpg.scale_noise(final_OU_noise_scale + (init_noise_scale - final_OU_noise_scale) * explr_pct_remaining)
    maddpg.reset_noise()
    maddpg.scale_beta(final_beta + (initial_beta - final_beta) * beta_pct_remaining)
    #for the duration of 100 episode with maximum length of 500 time steps
    time_step = 0
    kickable_counter = 0
    for et_i in range(0, episode_length):
        maddpg.reset_noise()

        # gather all the observations into a torch tensor 
        torch_obs = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                              requires_grad=False)
                     for i in range(maddpg.nagents)]
        # get actions as torch Variables
        torch_agent_actions = maddpg.step(torch_obs, explore=explore)
        # convert actions to numpy arrays
        agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
        # rearrange actions to be per environment
        
        # Get egreedy action
        params = np.asarray([ac[0][len(env.action_list):] for ac in agent_actions]) 
        actions = [[ac[i][:len(env.action_list)] for ac in agent_actions] for i in range(1)] # this is returning one-hot-encoded action for each agent 
        if explore:
            noisey_actions = [e_greedy(torch.tensor(a).view(1,len(env.action_list)),
                                                 eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)) for a in actions]     # get eps greedy action
        else:
            noisey_actions = [e_greedy(torch.tensor(a).view(1,len(env.action_list)),eps = 0) for a in actions]     # get eps greedy action


        # modify for multi agent
        randoms = noisey_actions[0][1]
        noisey_actions = [noisey_actions[0][0]]
        
        noisey_actions_for_buffer = [ac.data.numpy() for ac in noisey_actions]
        noisey_actions_for_buffer = np.asarray([ac[0] for ac in noisey_actions_for_buffer])

        
        obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents)]).T
        
        # use random unif parameters if e_greedy
        if randoms:
            noisey_actions_for_buffer = np.asarray([[val for val in (np.random.uniform(-1,1,3))]])
            params = np.asarray([[val for val in (np.random.uniform(-1,1,5))]])
        else:
            noisey_actions_for_buffer = np.asarray(actions[0])
            
        agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes 
        params_for_buffer = params

        actions_params_for_buffer = np.array([[np.concatenate((ac,pm),axis=0) for ac,pm in zip(noisey_actions_for_buffer,params_for_buffer)] for i in range(1)]).reshape(
            env.num_TA,env.action_params.shape[1] + len(env.action_list)) # concatenated actions, params for buffer

        # If kickable is True one of the teammate agents has possession of the ball
        kickable = False
        kickable = np.array([env.get_kickable_status(i,obs.T) for i in range(env.num_TA)]).any()
        if kickable == True:
            kickable_counter += 1


        _,_,d,world_stat = env.Step(agents_actions, 'team',params)
        # if d == True agent took an end action such as scoring a goal

        rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA) ])            

        next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents)]).T
        dones = np.hstack([env.d for i in range(env.num_TA)])
 

        print(next_obs)

        print(actions_params_for_buffer)
        print(dones)
        print(rewards)
        # Store variables for calculation of MC and n-step targets
        n_step_rewards.append(rewards)
        n_step_obs.append(obs)
        n_step_next_obs.append(next_obs)
        n_step_acs.append(actions_params_for_buffer)
        n_step_dones.append(dones)

        if (len(replay_buffer) >= batch_size and
            (t % steps_per_update) < 1) and t > burn_in_iterations:
            #if USE_CUDA:
            #    maddpg.prep_training(device='gpu')
            #else:
            maddpg.prep_training(device='cpu')
            for u_i in range(1):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(batch_size,
                                                  to_gpu=False,norm_rews=False)
                    maddpg.update(sample, a_i )
                maddpg.update_all_targets()
            maddpg.prep_rollouts(device='cpu')
            
            
        time_step += 1
        t += 1
        if t%1000 == 0:
            step_logger_df.to_csv('history.csv')
            
        # why don't we push the reward with gamma already instead of storing gammas?                     
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
                replay_buffer.push(n_step_ob, n_step_ac,n_step_rewards[n],n_step_next_ob,n_step_done,np.hstack([MC_target]),np.hstack([n_step_target])) 
                                       
            # log
            if time_step > 0 and ep_i > 1:
                step_logger_df = step_logger_df.append({'time_steps': time_step, 
                                                    'why': world_stat,
                                                    'kickable_percentages': (kickable_counter/time_step) * 100,
                                                    'average_reward': replay_buffer.get_average_rewards(time_step-1),
                                                   'cumulative_reward': replay_buffer.get_cumulative_rewards(time_step)}, 
                                                    ignore_index=True)
            break;  
        obs = next_obs

            #print(step_logger_df) 
        #if t%30000 == 0 and use_viewer:
        if t%30000 == 0 and use_viewer and ep_i > use_viewer_after:
            env._start_viewer()       

    #ep_rews = replay_buffer.get_average_rewards(time_step)

    #Saves Actor/Critic every particular number of episodes
    if ep_i%ep_save_every == 0 and ep_i != 0:
        #Saving the actor NN in local path, needs to be tested by loading
        if save_actor:
            ('Saving Actor NN')
            current_day_time = datetime.datetime.now()
            maddpg.save_actor('saved_NN/Actor/actor_' + str(current_day_time.month) + 
                                            '_' + str(current_day_time.day) + 
                                            '_'  + str(current_day_time.year) + 
                                            '_' + str(current_day_time.hour) + 
                                            ':' + str(current_day_time.minute) + 
                                            '_episode_' + str(ep_i) + '.pth')

        #Saving the critic in local path, needs to be tested by loading
        if save_critic:
            print('Saving Critic NN')
            maddpg.save_critic('saved_NN/Critic/critic_' + str(current_day_time.month) + 
                                            '_' + str(current_day_time.day) + 
                                            '_'  + str(current_day_time.year) + 
                                            '_' + str(current_day_time.hour) + 
                                            ':' + str(current_day_time.minute) + 
                                            '_episode_' + str(ep_i) + '.pth')



            
    

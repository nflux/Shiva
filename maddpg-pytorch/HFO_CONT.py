import itertools
import random
import numpy as np
import random 
import datetime
import os 
import csv
import itertools 
#import tensorflow.contrib.slim as slim
import numpy as np
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from torch import Tensor
import hfo
import time
import _thread as thread
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

from HFO_env import *

def zero_params(num_TA,params,action_index):
    for i in range(num_TA):
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

            
# default settings

action_level = 'low'
feature_level = 'low'

num_episodes = 100000
episode_length = 500 # FPS

replay_memory_size = 1000000
num_explore_episodes = 40  # Haus uses over 10,000 updates --
burn_in_iterations = 50000 # for time step
burn_in_episodes = float(burn_in_iterations)/episode_length
USE_CUDA = False 

final_noise_scale = 0.1
init_noise_scale = 1.00
steps_per_update = 1
untouched_time = 500

#Saving the NNs, currently set to save after each episode
save_critic = False
save_actor = False
#The NNs saved every #th episode.
ep_save_every = 20

#Load previous NNs, currently set to load only at initialization.
load_critic = False
load_actor = False

batch_size = 32
hidden_dim = int(1024)
a_lr = 0.00001 # actor learning rate
c_lr = 0.001 # critic learning rate
tau = 0.001 # soft update rate

t = 0
time_step = 0
kickable_counter = 0
n_training_threads = 8
explore = True
use_viewer = True

# D4PG atoms
gamma = 0.99
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
REWARD_STEPS = 5

# if using low level actions use non discrete settings
if action_level == 'high':
    discrete_action = True
else:
    discrete_action = False
    
    
if not USE_CUDA:
        torch.set_num_threads(n_training_threads)
        
env = HFO_env(num_TA=1, num_ONPC=0, num_trials = num_episodes, fpt = episode_length, 
              feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,offense_on_ball=False)

# if you want viewer
if use_viewer:
    env._start_viewer()

time.sleep(4)
print("Done connecting to the server ")

# initializing the maddpg 
maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=tau,
                                  a_lr=a_lr,
                                  c_lr=c_lr,
                                  hidden_dim=hidden_dim ,discrete_action=discrete_action,
                                  vmax=Vmax,vmin=Vmin,N_ATOMS=N_ATOMS,
                              REWARD_STEPS=REWARD_STEPS,DELTA_Z=DELTA_Z)



print('maddpg.nagents ', maddpg.nagents)
print('env.num_TA ', env.num_TA)  
print('env.num_features : ' , env.num_features)
#initialize the replay buffer of size 10000 for number of agent with their observations & actions 
replay_buffer = ReplayBuffer(replay_memory_size , env.num_TA,
                                 [env.num_features for i in range(env.num_TA)],
                                 [env.action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)])


    
reward_total = [ ]
num_steps_per_episode = []
end_actions = []
logger_df = pd.DataFrame()
step_logger_df = pd.DataFrame()

# for the duration of 1000 episodes 
for ep_i in range(0, num_episodes):

    n_step_rewards = []
    n_step_reward = 0
    n_step_obs = []
    n_step_acs = []

    maddpg.prep_rollouts(device='cpu')
    #define/update the noise used for exploration
    if ep_i < burn_in_episodes:
        explr_pct_remaining = 1.0
    else:
        explr_pct_remaining = max(0, num_explore_episodes - ep_i + burn_in_episodes) / (num_explore_episodes)
    maddpg.scale_noise(final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)
    maddpg.reset_noise()
    #for the duration of 100 episode with maximum length of 500 time steps
    time_step = 0
    kickable_counter = 0
    for et_i in range(0, episode_length):
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
            noisey_actions = [onehot_from_logits(torch.tensor(a).view(1,len(env.action_list)),
                                                 eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)) for a in actions]     # get eps greedy action
        else:
            noisey_actions = [onehot_from_logits(torch.tensor(a).view(1,len(env.action_list)),eps = 0) for a in actions]     # get eps greedy action

        noisey_actions_for_buffer = [ac.data.numpy() for ac in noisey_actions]
        noisey_actions_for_buffer = np.asarray([ac[0] for ac in noisey_actions_for_buffer])

        agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes 
        obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents)]).T

        params_for_buffer = zero_params(env.num_TA,params,agents_actions)

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



        # Store n-steps | send first and last to buffer with rewards = discounted sum rewards
        n_step_rewards.append(rewards)
        n_step_obs.append(obs)
        n_step_acs.append(actions_params_for_buffer)
        if et_i >= REWARD_STEPS:
            # TODO: reward stacking is probably wrong for multiagent
            # Get discounted reward sum for n steps
            for step in range(REWARD_STEPS):
                n_step_reward += n_step_rewards[-2 - step] * gamma**(REWARD_STEPS - 1 - step)
            # push first, last obs and rew sum
            replay_buffer.push(n_step_obs[-REWARD_STEPS], n_step_acs[-REWARD_STEPS], n_step_reward, next_obs, dones)
            n_step_reward = 0

   
        obs = next_obs

        time_step += 1

        t += 1
        if t%1000 == 0:
            step_logger_df.to_csv('history.csv')

        if (len(replay_buffer) >= batch_size and
            (t % steps_per_update) < 1) and t > burn_in_iterations:
            #if USE_CUDA:
            #    maddpg.prep_training(device='gpu')
            #else:
            maddpg.prep_training(device='cpu')
            for u_i in range(1):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(batch_size,
                                                  to_gpu=False,norm_rews=True)
                    #print('a_i ' , a_i )
                    maddpg.update(sample, a_i )
                maddpg.update_all_targets()
            maddpg.prep_rollouts(device='cpu')
        if d == True:
            step_logger_df = step_logger_df.append({'time_steps': time_step, 
                                                    'why': world_stat,
                                                    'kickable_percentages': (kickable_counter/time_step) * 100,
                                                    'average_reward': replay_buffer.get_average_rewards(time_step),
                                                   'cumulative_reward': replay_buffer.get_cumulative_rewards(time_step)}, 
                                                    ignore_index=True)

            # push rest n steps
            if et_i >= REWARD_STEPS and ep_i > 1:
                for n in range(REWARD_STEPS-1):
                    n_step_reward = 0
                    for step in range(REWARD_STEPS-1-n):
                        n_step_reward += n_step_rewards[-2 - step] * gamma**(REWARD_STEPS - 2 - step)
                    replay_buffer.push(n_step_obs[-REWARD_STEPS + 1 + n], n_step_acs[-REWARD_STEPS +1 + n], n_step_reward,obs,dones)                
            break;

            #print(step_logger_df) 
        #if t%30000 == 0 and use_viewer:
        if t%30000 == 0 and use_viewer and ep_i > 120:
            env._start_viewer()       

    ep_rews = replay_buffer.get_average_rewards(time_step)

    #Saves Actor/Critic every particular number of episodes
    if ep_i%ep_save_every == 0 and ep_i != 0:
        #Saving the actor NN in local path, needs to be tested by loading
        if save_actor:
            print('Saving Actor NN')
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



            
    
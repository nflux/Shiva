import itertools
import random
import numpy as np
import random 
#import tensorflow as tf
#import matplotlib.pyplot as plt 
#import scipy.misc
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



 # ./bin/HFO --offense-agents=1 --defense-npcs=0 --trials 170000 --frames-per-trial=100 --seed 123 --untouched-time=100 --record 
    
# default settings

action_level = 'low'
feature_level = 'low'

num_episodes = 100000
episode_length = 500 # FPS

replay_memory_size = 1000000
num_explore_episodes = 20  # Haus uses over 10,000 updates --
burn_in_iterations = 500000 # for time step
burn_in_episodes = float(burn_in_iterations)/episode_length
USE_CUDA = False 

final_noise_scale = 0.1
init_noise_scale = 1.00
steps_per_update = 1
untouched_time = 500

batch_size = 32
hidden_dim = int(1024)
a_lr = 0.00001
c_lr = 0.001
tau = 0.001

t = 0
time_step = 0
kickable_counter = 0
n_training_threads = 8
explore = True
use_viewer = True

# if using low level actions use non discrete settings
if action_level == 'high':
    discrete_action = True
else:
    discrete_action = False
    
    
if not USE_CUDA:
        torch.set_num_threads(n_training_threads)
        
env = HFO_env(num_TA=1, num_ONPC=0, num_trials = num_episodes, fpt = episode_length, 
              feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True)

# if you want viewer
if use_viewer:
    env._start_viewer()

time.sleep(2)
print("Done connecting to the server ")

# initializing the maddpg 
maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=tau,
                                  a_lr=a_lr,
                                  c_lr=c_lr,
                                  hidden_dim=hidden_dim ,discrete_action=discrete_action)



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
        
        #get the whole team observation 

        maddpg.prep_rollouts(device='cpu')
        #define the noise used for exploration
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
            
            
            actions_params_for_buffer = np.array([[ac[i] for ac in agent_actions] for i in range(1)]).reshape(
                env.num_TA,env.action_params.shape[1] + len(env.action_list)) # concatenated actions, params for buffer
            actions = [[ac[i][:len(env.action_list)] for ac in agent_actions] for i in range(1)] # this is returning one-hot-encoded action for each agent 
            if explore:
                noisey_actions = [onehot_from_logits(torch.tensor(a).view(1,3),eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining)) for a in actions]     # get eps greedy action
            else:
                noisey_actions = [onehot_from_logits(torch.tensor(a).view(1,3),eps = 0) for a in actions]     # get eps greedy action


            
            params = np.asarray([ac[0][len(env.action_list):] for ac in agent_actions])
            #print("ACTIONS: ",actions)
            agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in noisey_actions[0]] # convert the one hot encoded actions  to list indexes 

     
            obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents)]).T


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


        
            replay_buffer.push(obs, actions_params_for_buffer, rewards, next_obs, dones)
            obs = next_obs
            #for i in range(env.num_TA):
            #    logger_df = logger_df.append({'reward':env.Reward(i,'team')}, ignore_index=True)
            
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
                                                      to_gpu=False,norm_rews=False)
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
                break;
                
            
                
                #print(step_logger_df) 
            if t%30000 == 0 and use_viewer:
                env._start_viewer()       
           
        ep_rews = replay_buffer.get_average_rewards(time_step)
 #       print('episode rewards ', ep_rews )

import itertools
import random
import numpy as np
import random 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.misc
import os 
import csv
import itertools 
import tensorflow.contrib.slim as slim
import numpy as np

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



 # ./bin/HFO --offense-agents=1 --defense-npcs=0 --trials 20000 --frames-per-trial 1000 --seed 123

env = HFO_env(1,0,1,'left',False,1000,1000,'high','high')
time.sleep(0.1)
print("Done connecting to the server ")

# initializing the maddpg 
maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=0.08,
                                  lr=0.0002,
                                  hidden_dim=64)



print('maddpg.nagents ', maddpg.nagents)
print('env.num_TA ', env.num_TA)  
print('env.num_features : ' , env.num_features)
#initialize the replay buffer of size 10000 for number of agent with their observations & actions 
replay_buffer = ReplayBuffer(10000, env.num_TA,
                                 [env.num_features for i in range(env.num_TA)],
                                 [len(env.action_list) for i in range(env.num_TA)])

num_episodes = 20000
episode_length = 1000
t = 0

env.Step([random.randint(0,len(env.action_list)-1) for i in range(env.num_TA)],'team')
reward_total = [ ]
logger_df = pd.DataFrame({'reward':reward_total})
# for the duration of 1000 episodes 
for ep_i in range(0, num_episodes):
        
        #get the whole team observation 
        obs = np.asarray(env.team_obs)
        
        maddpg.prep_rollouts(device='cpu')
        #define the noise used for exploration
        explr_pct_remaining = max(0, num_episodes - ep_i) / num_episodes
        maddpg.scale_noise(0 + (0.3 - 0.0) * explr_pct_remaining)
        maddpg.reset_noise()
        #for the duration of 100 episode with maximum length of 500 time steps 
        

            
        for et_i in range(episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            

            
            #print('env team observation ',env.team_obs) 
            #print("Agent 0 Observation:",env.Observation(0,'team'))
            #print("Reward:",env.Reward(0,'team'))
            
            # gather all the observations into a torch tensor 
            torch_obs = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            #print('vstack obs: ' , np.vstack(obs[0,:]))
            # get actions as torch Variables
            #print('torch_obs', torch_obs)
            # feed the maddpg with the obsevation of all agents 
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(1)] # this is returning one-hot-encoded action for each agent 
            agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in actions[0]] # convert the one hot encoded actions  to list indexes 

            #print("Actions taken: ", agent_actions)
            obs =  np.vstack([env.Observation(i,'team') for i in range(maddpg.nagents)] ) 

            
            
            env.Step(agents_actions, 'team') # take the fucking actions
            
            
            
            #rewards = np.vstack([env.Reward(i,'team') for i in range(env.num_TA) ])
            rewards = np.vstack([env.Reward(i,'team') for i in range(env.num_TA) ])
            #q('rewards ',  rewards)
            #next_obs = np.asarray(env.team_obs)
            
            next_obs = np.vstack([env.Observation(i,'team') for i in range(maddpg.nagents)] )

            dones = np.vstack([0 for i in range(env.num_TA)])
            print(rewards)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            for i in range(env.num_TA):
                logger_df = logger_df.append({'reward':env.Reward(i,'team')}, ignore_index=True)
            
            t += 1
            if t%1000 == 0:
                    logger_df.to_csv('history.csv')
            if (len(replay_buffer) >= 32 and
                (t % 10) < 1):
                #if USE_CUDA:
                #    maddpg.prep_training(device='gpu')
                #else:
                maddpg.prep_training(device='cpu')
                for u_i in range(1):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(32,
                                                      to_gpu=False,norm_rews=False)
                        #print('sample: ', sample)
                        #print('a_i ' , a_i )
                        maddpg.update(sample, a_i )
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
                logger_df
        ep_rews = replay_buffer.get_average_rewards(episode_length)
 #       print('episode rewards ', ep_rews )
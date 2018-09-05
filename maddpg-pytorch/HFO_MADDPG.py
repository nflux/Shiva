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




env = HFO_env(1,0,0,'left',False,1000,1000,'high','high')
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

t = 0
# for the duration of 10 episodes 
for ep_i in range(0, 10):
        
        #get the whole team observation 
        obs = np.asarray(env.team_obs)
        
        maddpg.prep_rollouts(device='cpu')
        #define the noise used for exploration
        explr_pct_remaining = max(0, 20 - ep_i) / 20
        maddpg.scale_noise(0 + (0.3 - 0.0) * explr_pct_remaining)
        maddpg.reset_noise()
        #for the duration of 1 episode with maximum length of 10 time steps 
        for et_i in range(10):
            
            # rearrange observations to be per agent, and convert to torch Variable
            
            # without this initial random step the agents seemed to crash, it's a waste of 1 timestep per episode 
            env.Step([random.randint(0,3) for i in range(env.num_TA)],'team')
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
            torch_agent_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            print('agent_actions ', agent_actions)
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(1)] # this is returning one-hot-encoded action for each agent 
            print('actions, ', actions)
            agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in actions[0]] # convert the one hot encoded actions  to list indexes 

            obs =  np.vstack([env.Observation(i,'team') for i in range(maddpg.nagents)] ) 

            env.Step(agents_actions, 'team') # take the fucking actions
            
            
            
            #rewards = np.vstack([env.Reward(i,'team') for i in range(env.num_TA) ])
            rewards = np.vstack([10 for i in range(env.num_TA) ])
            #print('rewards ',  rewards)
            #next_obs = np.asarray(env.team_obs)
            next_obs = np.vstack([env.Observation(i,'team') for i in range(maddpg.nagents)] )

            dones = np.vstack([0 for i in range(env.num_TA)])
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            
            t += 1
            if (len(replay_buffer) >= 32 and
                (t % 10) < 1):
                #if USE_CUDA:
                #    maddpg.prep_training(device='gpu')
                #else:
                maddpg.prep_training(device='cpu')
                for u_i in range(1):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(2,
                                                      to_gpu=False)
                        print('sample: ', sample)
                        print('a_i ' , a_i )
                        #maddpg.update(sample, a_i )
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(100)
        print('episode rewards ', ep_rews )
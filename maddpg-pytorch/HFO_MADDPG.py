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
print('')



print("Done connecting to the server ")

# initializing the maddpg 
maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=0.08,
                                  lr=0.0002,
                                  hidden_dim=64)

#initialize the replay buffer of size 10000 for 1 agent with 12 observations and 4 actions 
replay_buffer = ReplayBuffer(10000, 1,
                                 [12],
                                 [4])

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
            #print('maddpg.nagents ', maddpg.nagents)
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
            actions = [ac[0] for ac in agent_actions] # is this returning one-hot-encoded for each agent ? 
            print('actions, ', actions)
            print('best actions ', actions.index(max(actions)))
            env.Step([0], 'team')
            
            '''rewards = np.asarray([env.Reward(i,'team') for i in range(env.num_TA) ]).T
            next_obs = np.asarray(env.team_obs)
            dones = np.asarray([0 for i in range(env.num_TA)])
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs'''
            
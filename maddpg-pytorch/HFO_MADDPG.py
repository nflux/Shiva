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
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG

from HFO_env import *

#matplotlib inline

#from helper import * 


env = HFO_env(8,0,1,'left',False,1000,1000,'high','high') 


maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=0.08,
                                  lr=0.0002,
                                  hidden_dim=64)

replay_buffer = ReplayBuffer(10000, 1,
                                 [12],
                                 [4])
t = 0
for ep_i in range(0, 1000):
        
        
        obs = np.asarray(env.team_obs)
         
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, 20 - ep_i) / 20
        maddpg.scale_noise(0 + (0.3 - 0.0) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(1000):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [ac[0] for ac in agent_actions]
            print('actions, ', actions)
            #env.Step(actions, 'team')
            #next_obs, rewards, dones, infos = env.Step(actions) ****what is info?

            next_obs, rewards, dones = env.Step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            
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
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy,zero_params,pretrain_process,prep_session
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
from rc_env import *
from trainer import launch_eval
import _thread as thread

eval_episodes = 50
session_path = "training_sessions/eval/"
load_path = session_path +"ensemble_models/"
eval_log_dir = session_path +"eval_log" # evaluation logfiles
eval_hist_dir = session_path +"eval_history"
num_TA =3
num_OA =3
port = 63000
episode_length = 500
device = "cuda"
use_viewer = True
launch_eval(
    ["agent2d/agent2D.pth" for i in range(num_TA)], # models directory -> agent -> most current episode
    eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation",
    port,num_TA,num_OA,episode_length,device,use_viewer)
'''launch_eval(
    ['1v1/model_%i.pth' % i for i in range(num_TA)], # models directory -> agent -> most current episode
    eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation",
    7000,num_TA,num_OA,episode_length,device,use_viewer)'''
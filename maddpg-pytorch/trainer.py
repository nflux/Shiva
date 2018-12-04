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
from HFO import hfo
import time
import _thread as thread
import argparse
import torch
from pathlib import Path
import argparse
from torch.autograd import Variable#from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from evaluation_env import *


def launch_eval(filenames,eval_episodes = 10,log_dir = "eval",log='eval',port=7000,num_TNPC = 0,num_TA=1,num_OA=0, num_ONPC=0, fpt = 500,device="cpu"):
    
    env = evaluation_env(num_TNPC = 0,num_TA=num_TA,num_OA=0, num_ONPC=num_ONPC, num_trials = eval_episodes, fpt = fpt,feat_lvl = 'low', act_lvl = 'low',
                         untouched_time = 500,fullstate=True,offense_on_ball=False,
                         port=port,log_dir=log_dir)
    time.sleep(2.0)
    maddpg = MADDPG.init_from_save_evaluation(filenames,num_TA)
    time.sleep(1.5)

    team_step_logger_df = pd.DataFrame()
    env.launch()
    time.sleep(1.5)
    maddpg.scale_noise(0.0)
    maddpg.reset_noise()
    team_kickable_counter = 0
    t = 0
    time_step = 0
    maddpg.prep_training(device=device) # GPU for forward passes?? 

    env._start_viewer()
    # launch evaluation episodes
    for ep_i in range(eval_episodes):
        for et_i in range(fpt):
            torch_obs_team = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                                    requires_grad=False)
                            for i in range(maddpg.nagents_team)]
            team_torch_agent_actions,_ = maddpg.step(torch_obs_team,torch_obs_team, explore=False)
            team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
            team_params = np.asarray([ac[0][len(env.action_list):] for ac in team_agent_actions]) 
            team_actions = [[ac[0][:len(env.action_list)] for ac in team_agent_actions]]

            tensors = []
            rands = []


            team_noisey_actions = [e_greedy(torch.tensor(a).view(env.num_TA,len(env.action_list)), env.num_TA, eps = 0.000001) for a in team_actions]

            team_randoms = [team_noisey_actions[0][1][i] for i in range(env.num_TA)]

            team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T

            team_noisey_actions_for_buffer = np.asarray([[val for val in (np.random.uniform(-1,1,3))] if ran else action for ran,action in zip(team_randoms,team_actions[0])])
            team_params = np.asarray([[val for val in (np.random.uniform(-1,1,5))] if ran else p for ran,p in zip(team_randoms,team_params)])

            team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes

            kickable = False
            kickable = np.array([env.get_kickable_status(i,team_obs.T) for i in range(env.num_TA)]).any()
            if kickable == True:
                team_kickable_counter += 1


            _,_,d,world_stat = env.Step(team_agents_actions,team_agents_actions,team_params,team_params)
            team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])

            team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            team_done = env.d
            time_step += 1
            t += 1
            if d == True:
                team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                        'why': env.team_envs[0].statusToString(world_stat),
                                                        'kickable_percentages': (team_kickable_counter/time_step) * 100,
                                                        'goals_scored': env.scored_counter_left/env.num_TA}, 
                                                        ignore_index=True)

                
                break;  
            team_obs =  team_next_obs
                
    team_step_logger_df.to_csv('%s.csv' % log)

    
#launch_eval(['models/2_vs_2/time_12_4_9/model_episode_2_agent_0.pth','models/2_vs_2/time_12_4_9/model_episode_2_agent_1.pth'],eval_episodes = 10,log_dir = "eval",log='eval',port=6000,num_TNPC = 0,num_TA=2,num_OA=0, num_ONPC=0, fpt = 500,device="cuda")

#launch_eval(filenames,eval_episodes,log_dir = log_dir,log=history,port=port,num_TNPC = 0,num_TA=num_TA,num_OA=0, num_ONPC=num_ONPC, fpt = 500,device=device)


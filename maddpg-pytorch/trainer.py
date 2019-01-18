import itertools
import random
import datetime
import os 
import csv
import itertools 
#import tensorflow.contrib.slim as slim
import numpy as np
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy_bool,zero_params,pretrain_process
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
import subprocess

def launch_eval(filenames,eval_episodes = 10,log_dir = "eval_log",log='eval',port=7000,
                num_TA=1,num_ONPC=0, fpt = 500,device="cpu",use_viewer=False,goalie=False):


    control_rand_init = True
    ball_x_min = -0.1
    ball_x_max = 0.1
    ball_y_min = -0.1
    ball_y_max = 0.1
    agents_x_min = -0.3
    agents_x_max = 0.3
    agents_y_min = -0.3
    agents_y_max = 0.3
    change_every_x = 1000000000
    change_agents_x = 0.01
    change_agents_y = 0.01
    change_balls_x = 0.01
    change_balls_y = 0.01
    print('killing the evaluation server from inside the thread')

    print(eval_episodes)
    subprocess.Popen("ps -ef | grep 7000 | awk '{print $2}' | xargs kill",shell=True)
    time.sleep(1)
    env = evaluation_env(num_TNPC = 0,num_TA=num_TA,num_OA=0, num_ONPC=num_ONPC, num_trials = eval_episodes, fpt = fpt,feat_lvl = 'low', act_lvl = 'low',
                         untouched_time = 500,fullstate=True,offense_on_ball=False,
                         port=port,log_dir=log_dir,record=False,goalie=goalie,agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init)
    time.sleep(2.0)
    maddpg = MADDPG.init_from_save_evaluation(filenames,num_TA)
    time.sleep(1.5)

    team_step_logger_df = pd.DataFrame()
    env.launch()
    time.sleep(1.5)
    maddpg.scale_noise(0.0)
    maddpg.reset_noise()
    t = 0
    maddpg.prep_training(device=device) # GPU for forward passes?? 

    if use_viewer:
        env._start_viewer()
    # launch evaluation episodes
    for ep_i in range(eval_episodes):
        time_step = 0
        team_kickable_counter = 0
        for et_i in range(fpt):
            torch_obs_team = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                                    requires_grad=False)
                            for i in range(maddpg.nagents_team)]
  
            tensors = []
            rands = []

        # Get e-greedy decision
    
            team_randoms = e_greedy_bool(env.num_TA,eps = 0,device=device)
            opp_randoms = e_greedy_bool(env.num_OA,eps = 0,device=device)
            team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_team,team_randoms,opp_randoms,explore=False) # leave off or will gumbel sample
            team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
            team_params = np.asarray([ac[0][len(env.action_list):] for ac in team_agent_actions]) 

            team_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in team_agent_actions]])

            team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T

            team_noisey_actions_for_buffer = team_actions[0]

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
    #env.kill_viewer()
    

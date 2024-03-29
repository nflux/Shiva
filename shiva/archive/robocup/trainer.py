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
# from utils.buffer import ReplayBuffer
#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from evaluation_env import *
import subprocess
import pandas as pd
def launch_eval(filenames,eval_episodes = 10,log_dir = "eval_log",log='eval',port=7000,
                num_TA=1,num_ONPC=0, fpt = 500,device="cpu",use_viewer=False,goalie=True):

    if os.path.isdir(os.getcwd() + '/log_' + str(port)):
        file_list = os.listdir(os.getcwd() + '/log_' + str(port))
        [os.remove(os.getcwd() + '/log_' + str(port) + '/' + f) for f in file_list]
    else:
        os.mkdir(os.getcwd() + '/log_' + str(port))
    
    if os.path.isdir(os.getcwd() + '/pt_logs_' + str(port)):
        file_list = os.listdir(os.getcwd() + '/pt_logs_' + str(port))
        [os.remove(os.getcwd() + '/pt_logs_' + str(port) + '/' + f) for f in file_list]
    else:
        os.mkdir(os.getcwd() + '/pt_logs_' + str(port))


    control_rand_init = True
    ball_x_min = -0.1
    ball_x_max = -0.1
    ball_y_min = -0.2
    ball_y_max = -0.2
    agents_x_min = -0.3
    agents_x_max = 0.3
    agents_y_min = -0.3
    agents_y_max = 0.3
    change_every_x = 1000000000
    change_agents_x = 0.01
    change_agents_y = 0.01
    change_balls_x = 0.01
    change_balls_y = 0.01
    ag1 = pd.read_csv("pt_logs_2000/log_actions_left_1.csv",header=None)
    ag2 = pd.read_csv("pt_logs_2000/log_actions_left_2.csv",header=None)
    ag3 = pd.read_csv("pt_logs_2000/log_actions_left_3.csv",header=None)
    ob1 = pd.read_csv("pt_logs_2000/log_obs_left_1.csv",header=None)
    ob2 = pd.read_csv("pt_logs_2000/log_obs_left_2.csv",header=None)
    ob3 = pd.read_csv("pt_logs_2000/log_obs_left_3.csv",header=None)
    stat = pd.read_csv("pt_logs_2000/log_status.csv",header=None)
    start1 = np.where(ag1.iloc[:,0])[0][0]
    start2 = np.where(ag2.iloc[:,0])[0][0]
    start3 = np.where(ag3.iloc[:,0])[0][0]
    ep_length = (np.where(stat.iloc[:,1])[0][0] + 1)
    manual_feed_actions = True
    manual_feed_obs = False
    defense_team_bin = 'base'
    print(eval_episodes)
    #subprocess.Popen("ps -ef | grep 7000 | awk '{print $2}' | xargs kill",shell=True)
    time.sleep(1)
    env = evaluation_env(num_TNPC = 0,num_TA=num_TA,num_OA=0, num_ONPC=num_ONPC, num_trials = eval_episodes, fpt = fpt,feat_lvl = 'simple', act_lvl = 'low',
                         untouched_time = 500,fullstate=True,ball_x_min = ball_x_min,ball_x_max =ball_x_max,
                         ball_y_min = ball_y_min,ball_y_max = ball_y_max,offense_on_ball=False,
                         port=63000,log_dir=log_dir,rcss_log_game=False,hfo_log_game=False,record=False,goalie=goalie,agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record_server=True,defense_team_bin=defense_team_bin)
    time.sleep(2.0)
    maddpg = MADDPG.init_from_save(filenames,num_TA)
    time.sleep(1.5)
    LSTM_policy = True
    team_step_logger_df = pd.DataFrame()
    env.launch()
    time.sleep(1.5)
    maddpg.scale_noise(0.0)
    maddpg.reset_noise()
    t = 0
    maddpg.prep_training(device=device) # GPU for forward passes?? 
    maddpg.prep_policy_rollout(device=device)
    use_viewer = True
    if use_viewer:
        env._start_viewer()
    # launch evaluation episodes
    first_action = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    for ep_i in range(eval_episodes):
        
        time_step = 0
        team_kickable_counter = 0
        if LSTM_policy:
            maddpg.zero_hidden_policy(1,maddpg.torch_device)
        for et_i in range(fpt):
            torch_obs_team = [Variable(torch.Tensor(np.vstack(env.Observation(i,'team')).T),
                                    requires_grad=False)
                            for i in range(maddpg.nagents_team)]
            if manual_feed_obs:
                if et_i == 0:
                    torch_obs_team = [Variable(torch.from_numpy(np.vstack(np.concatenate((ob1.iloc[ep_length+et_i].values[1:],first_action),axis=0)).T).float(),requires_grad=False),Variable(torch.from_numpy(np.vstack(np.concatenate((ob2.iloc[ep_length+et_i].values[1:],first_action),axis=0)).T).float(),requires_grad=False),Variable(torch.from_numpy(np.vstack(np.concatenate((ob3.iloc[ep_length+et_i].values[1:],first_action),axis=0)).T).float(),requires_grad=False)]
                else:
                    ag1_lac = ag1.iloc[start1+ep_length+et_i-1].values[1:]
                    ag2_lac = ag2.iloc[start2+ep_length+et_i-1].values[1:]
                    ag3_lac = ag3.iloc[start3+ep_length+et_i-1].values[1:]
                    torch_obs_team = [Variable(torch.from_numpy(np.vstack(np.concatenate((ob1.iloc[ep_length+et_i].values[1:],ag1_lac),axis=0)).T).float(),requires_grad=False),Variable(torch.from_numpy(np.vstack(np.concatenate((ob2.iloc[ep_length+et_i].values[1:],ag2_lac),axis=0)).T).float(),requires_grad=False),Variable(torch.from_numpy(np.vstack(np.concatenate((ob3.iloc[ep_length+et_i].values[1:],ag3_lac),axis=0)).T).float(),requires_grad=False)]
                    
            tensors = []
            rands = []

        # Get e-greedy decision
    
            team_randoms = e_greedy_bool(env.num_TA,eps = 0,device=device)
            opp_randoms = e_greedy_bool(env.num_OA,eps = 0,device=device)

            team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_team,team_randoms,opp_randoms,explore=False,parallel=False) # leave off or will gumbel sample
            team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
            if manual_feed_actions:
                team_agent_actions = [np.array([ag1.iloc[start1+ep_length+et_i].values[1:]]),np.array([ag2.iloc[start2+et_i+ep_length].values[1:]]),np.array([ag3.iloc[start3+ep_length +et_i].values[1:]])]
                
            team_params = np.asarray([ac[0][len(env.action_list):] for ac in team_agent_actions]) 
            #if et_i==0:
                #print(ag2.iloc[start2 + ep_length])
            team_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in team_agent_actions]])


            #team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T

            team_noisey_actions_for_buffer = team_actions[0]

            team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes

            kickable = False
            #kickable = np.array([env.get_kickable_status(i,team_obs.T) for i in range(env.num_TA)]).any()
            #if kickable == True:
            #    team_kickable_counter += 1


            _,_,d,world_stat = env.Step(team_agents_actions,team_agents_actions,team_params,team_params,team_agent_actions,team_agent_actions)
            team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])

            team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            team_done = env.d
            time_step += 1
            t += 1
            if d == True:
                t += 0 
                team_step_logger_df = team_step_logger_df.append({'time_steps': time_step,
                                                        'why': env.team_envs[0].statusToString(world_stat),
                                                        'kickable_percentages': (team_kickable_counter/time_step) * 100,
                                                        'goals_scored': env.scored_counter_left/env.num_TA}, 
                                                        ignore_index=True)

                
                break;  
            team_obs =  team_next_obs
            
                
    team_step_logger_df.to_csv('%s.csv' % log)
    #env.kill_viewer()
    

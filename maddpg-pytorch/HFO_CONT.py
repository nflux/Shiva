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
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy,zero_params,pretrain_process,prep_session,e_greedy_bool
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
from HFO_env import *
from trainer import launch_eval
import torch.multiprocessing as mp
import _thread as thread
import dill

def update_thread(agentID,to_gpu,buffer_size,batch_size,team_replay_buffer,opp_replay_buffer,critic_mod_act,critic_mod_obs,number_of_updates,maddpg_pick,
                            load_path,ep_i,ensemble_path,current_ensembles,forward_pass):

    maddpg = dill.loads(maddpg_pick)
    
    for _ in range(number_of_updates):
        inds = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        team_sample = team_replay_buffer.sample(inds,to_gpu=to_gpu,norm_rews=False)
        opp_sample = opp_replay_buffer.sample(inds,to_gpu=to_gpu,norm_rews=False)

        maddpg.update_centralized_critic(team_sample, opp_sample, agentID, 'team',
                            act_only=critic_mod_act, obs_only=critic_mod_obs,forward_pass=forward_pass)
        maddpg.update_agent_targets(agentID)
    maddpg.save(load_path,ep_i)
    maddpg.save_ensembles(ensemble_path,current_ensembles)


if __name__ == "__main__":  
    mp.set_start_method('spawn',force=True)

    # Parseargs --------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Load port and log directory')
    parser.add_argument('-port', type=int,default=6000,
                    help='An integer for port number')
    parser.add_argument('-log_dir', type=str, default='log',
                    help='A name for log directory')
    parser.add_argument('-log', type=str, default='history',
                    help='A name for log file ')

    args = parser.parse_args()
    log_dir = args.log_dir
    port = args.port
    history = args.log
    # --------------------------------------
        
    # options ------------------------------
    action_level = 'low'
    feature_level = 'low'
    USE_CUDA = True
    if USE_CUDA:
        device = 'cuda'
        to_gpu = True
    else:
        to_gpu = False
        device = 'cpu'

    use_viewer = False
    use_viewer_after = 1000 # If using viewer, uses after x episodes
    n_training_threads = 8
    rcss_log_game = False #Logs the game using rcssserver
    hfo_log_game = False #Logs the game using HFO
    # default settings ---------------------
    num_episodes = 10000000
    replay_memory_size = 100000
    episode_length = 500 # FPS
    untouched_time = 200
    burn_in_iterations = 500 # for time step
    burn_in_episodes = float(burn_in_iterations)/untouched_time
    train_team = True
    train_opp = False
    # --------------------------------------
    # Team ---------------------------------
    num_TA = 2
    num_OA = 2
    num_TNPC = 0
    num_ONPC = 0
    offense_team_bin='helios10'
    defense_team_bin='helios11'  
    goalie = False
    team_rew_anneal_ep = 1500 # reward would be
    # hyperparams--------------------------
    batch_size = 256
    hidden_dim = int(512)
    a_lr = 0.0001 # actor learning rate
    c_lr = 0.001 # critic learning rate
    tau = 0.001 # soft update rate
    steps_per_update = 2500
    number_of_updates = 500
    # exploration --------------------------
    explore = True
    final_OU_noise_scale = 0.1
    final_noise_scale = 0.1
    init_noise_scale = 1.00
    num_explore_episodes = 500 # Haus uses over 10,000 updates --
    # --------------------------------------
    #D4PG Options --------------------------
    D4PG = True
    gamma = 0.99 # discount
    Vmax = 35
    Vmin = -35
    N_ATOMS = 25
    DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
    n_steps = 1
    # n-step update size 
    # Mixed taqrget beta (0 = 1-step, 1 = MC update)
    initial_beta = 0.4
    final_beta = 0.4
    num_beta_episodes = 1000000000
    #---------------------------------------
    #TD3 Options ---------------------------
    TD3 = True
    TD3_delay_steps = 2
    TD3_noise = 0.01
    # -------------------------------------- 
    #Pretrain Options ----------------------
    # To use imitation exporation run N TNPC vs N ONPC for the desired number of episodes
    # Copy the base_left-11.log and -7.log (for 2v2)  to Pretrain_Files and rerun this file.
    # (Also we must delete all the "garbage" at the beginning of the log files. The first line should be the second instance of 0 4 M StateFeatures)
    Imitation_exploration = False
    test_imitation = False  # After pretrain, infinitely runs the current pretrained policy
    pt_critic_updates = 30000
    pt_actor_updates = 30000
    pt_actor_critic_updates = 0
    pt_imagination_branch_pol_updates = 100
    pt_episodes = 1000# num of episodes that you observed in the gameplay between npcs
    pt_timesteps = 125000# number of timesteps to load in from files
    pt_EM_updates = 300
    pt_beta = 1.0
    #---------------------------------------
    #I2A Options ---------------------------
    I2A = False
    decent_EM = False
    EM_lr = 0.005
    obs_weight = 10.0
    rew_weight = 1.0
    ws_weight = 1.0
    rollout_steps = 1
    LSTM_hidden=16
    imagination_policy_branch = True
    #---------------------------------------
    # Self-Imitation Learning Options ------
    SIL = False
    SIL_update_ratio = 1
    #---------------------------------------
    #Critic Input Modification 
    critic_mod = True
    # NOTE: When both are False but critic_mod is true the critic takes both
    # actions and observations from the opposing side
    critic_mod_act = False
    critic_mod_obs = False
    critic_mod_both = ((critic_mod_act == False) and (critic_mod_obs == False) and critic_mod)
    #---------------------------------------
    # Control Random Initilization of Agents and Ball
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
    # Self-play ----------------------------
    load_random_nets = True
    load_random_every = 25
    k_ensembles = 1
    current_ensembles = [0]*num_TA # initialize which ensembles we start with
    self_play_proba = 0.8
    # --------------------------------------
    #Save/load -----------------------------
    save_nns = True
    ep_save_every = 25 # episodes
    load_nets = False # load previous sessions' networks from file for initialization
    initial_models = ["training_sessions/1_11_8_1_vs_1/ensemble_models/ensemble_agent_0/model_0.pth"]
    first_save = True # build model clones for ensemble
    # --------------------------------------
    # Evaluation ---------------------------
    evaluate = False
    eval_after = 500
    eval_episodes = 11
    # --------------------------------------
    # LSTM -------------------------------------------
    LSTM = False
    LSTM_PC = False # PC (Policy & Critic)
    if LSTM and LSTM_PC:
        print('Only one LSTM flag can be True or both False')
        exit(0)
    if LSTM or LSTM_PC:
        trace_length = 20
    else:
        trace_length = 0
    hidden_dim_lstm = 512
    # -------------------------------------------------
    # Optimization ------------------------------------
    parallel_process = True
    forward_pass = True
    # -------------------------------------------------
    # Prep Session Files ------------------------------
    session_path = None
    current_day_time = datetime.datetime.now()
    session_path = 'training_sessions/' + \
                                    str(current_day_time.month) + \
                                    '_' + str(current_day_time.day) + \
                                    '_' + str(current_day_time.hour) + '_' + \
                                    str(num_TA) + '_vs_' + str(num_OA) + "/"
    hist_dir = session_path +"history"
    eval_hist_dir = session_path +"eval_history"
    eval_log_dir = session_path +"eval_log" # evaluation logfiles
    load_path = session_path +"models/"
    ensemble_path = session_path +"ensemble_models/"
    prep_session(session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,log_dir,num_TA) # Generates directories and files for the session

    # --------------------------------------
    # initialization -----------------------
    t = 0
    time_step = 0
    # if using low level actions use non discrete settings
    if action_level == 'high':
        discrete_action = True
    else:
        discrete_action = False
    if not USE_CUDA:
            torch.set_num_threads(n_training_threads)

    #Initialization for either M vs N, M vs 0, or N vs 0
    if num_TA > 0:
        has_team_Agents = True
    else:
        has_team_Agents = False

    if num_OA > 0:
        has_opp_Agents = True
    else:
        has_opp_Agents = False   

    env = HFO_env(num_TNPC = num_TNPC,num_TA=num_TA,num_OA=num_OA, num_ONPC=num_ONPC, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length, # create environment
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=port,log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game, team_rew_anneal_ep=team_rew_anneal_ep,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=True,
                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin)

    #The start_viewer here is added to automatically start viewer with npc Vs npcs
    if num_TNPC > 0 and num_ONPC > 0:
        env._start_viewer() 

    time.sleep(3)
    print("Done connecting to the server ")

    # initializing the maddpg 
    if load_nets:
        maddpg = MADDPG.init_from_save_evaluation(initial_models,num_TA) # from evaluation method just loads the networks
    else:
        maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                adversary_alg= "MADDPG",device=device,
                                gamma=gamma,batch_size=batch_size,
                                tau=tau,
                                a_lr=a_lr,
                                c_lr=c_lr,
                                hidden_dim=hidden_dim ,discrete_action=discrete_action,
                                vmax=Vmax,vmin=Vmin,N_ATOMS=N_ATOMS,
                                n_steps=n_steps,DELTA_Z=DELTA_Z,D4PG=D4PG,beta=initial_beta,
                                TD3=TD3,TD3_noise=TD3_noise,TD3_delay_steps=TD3_delay_steps,
                                I2A = I2A, EM_lr = EM_lr,
                                obs_weight = obs_weight, rew_weight = rew_weight, ws_weight = ws_weight, 
                                rollout_steps = rollout_steps,LSTM_hidden=LSTM_hidden,decent_EM = decent_EM,
                                imagination_policy_branch = imagination_policy_branch,critic_mod_both=critic_mod_both,
                                critic_mod_act=critic_mod_act, critic_mod_obs= critic_mod_obs,
                                LSTM=LSTM, LSTM_PC=LSTM_PC, trace_length=trace_length, hidden_dim_lstm=hidden_dim_lstm) 


    team_replay_buffer = ReplayBuffer(replay_memory_size , env.num_TA, episode_length,
                                        [env.team_num_features for i in range(env.num_TA)],
                                    [env.team_action_params.shape[1] + len(env.action_list) for i in range(env.num_TA)], 
                                    batch_size, LSTM, LSTM_PC)

    #Added to Disable/Enable the opp agents
    if has_opp_Agents:
        #initialize the replay buffer of size 10000 for number of opponent agent with their observations & actions 
        opp_replay_buffer = ReplayBuffer(replay_memory_size , env.num_OA, episode_length,
                                    [env.opp_num_features for i in range(env.num_OA)],
                                    [env.opp_action_params.shape[1] + len(env.action_list) for i in range(env.num_OA)], 
                                    batch_size, LSTM, LSTM_PC)

    # If using critic mod the opponent buff will be less than the team buff
    if critic_mod_both and has_opp_Agents:
        ptr_buff = opp_replay_buffer
    else:
        ptr_buff = team_replay_buffer

    reward_total = [ ]
    num_steps_per_episode = []
    end_actions = [] 
    # logger_df = pd.DataFrame()
    team_step_logger_df = pd.DataFrame()
    opp_step_logger_df = pd.DataFrame()


    # -------------------------------------
    # PRETRAIN ############################
    if Imitation_exploration:

        team_files = ['Pretrain_Files/3v3_CentQ/base_left-11.log','Pretrain_Files/3v3_CentQ/base_left-7.log','Pretrain_Files/3v3_CentQ/base_left-8.log','Pretrain_Files/3v3_CentQ/base_right-2.log','Pretrain_Files/3v3_CentQ/base_right-3.log','Pretrain_Files/3v3_CentQ/base_right-4.log']
        #opp_files = ['Pretrain_Files/base_left-1.log','Pretrain_Files/base_left-2.log']

        team_pt_obs, team_pt_status,team_pt_actions,opp_pt_obs, opp_pt_status,opp_pt_actions = pretrain_process(fnames = team_files,timesteps = pt_timesteps,num_features = env.team_num_features)


        print("Length of team obs,stats,actions",len(team_pt_obs),len(team_pt_status),len(team_pt_actions))
        print("Length of opp obs,stats,actions",len(opp_pt_obs),len(opp_pt_status),len(opp_pt_actions))

        ################## Base Left #########################
        pt_time_step = 0
        for ep_i in range(0, pt_episodes):
            if ep_i % 100 == 0:
                print("Pushing Pretrain Base-Left Episode:",ep_i)
                
            
            # team n-step
            team_n_step_rewards = []
            team_n_step_obs = []
            team_n_step_acs = []
            n_step_next_obs = []
            team_n_step_dones = []
            team_n_step_ws = []

            #define/update the noise used for exploration
            explr_pct_remaining = 0.0
            beta_pct_remaining = 0.0
            maddpg.scale_noise(0.0)
            maddpg.reset_noise()
            maddpg.scale_beta(pt_beta)
            d = False
            
            for et_i in range(0, episode_length):            
                world_stat = team_pt_status[pt_time_step]
                d = False
                if world_stat != 0.0:
                    d = True

                #### Team ####
                team_n_step_acs.append(team_pt_actions[pt_time_step])
                team_n_step_obs.append(np.array(team_pt_obs[pt_time_step]).T)
                team_n_step_ws.append(world_stat)
                n_step_next_obs.append(np.array(team_pt_obs[pt_time_step+1]).T)
                team_n_step_rewards.append(np.hstack([env.getPretrainRew(world_stat,d,"base_left") for i in range(env.num_TA) ]))          
                team_n_step_dones.append(d)

                # Store variables for calculation of MC and n-step targets for team
                pt_time_step += 1
                if d == True: # Episode done
                    # Calculate n-step and MC targets
                    for n in range(et_i+1):
                        MC_targets = []
                        n_step_targets = []
                        for a in range(env.num_TA):
                            MC_target = 0
                            n_step_target = 0
                            
                            for step in range(et_i+1 - n): # sum MC target
                                MC_target += team_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
                            MC_targets.append(MC_target)
                            if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                                for step in range(n_steps): 
                                    n_step_target += team_n_step_rewards[n + step][a] * gamma**(step)
                                n_step_targets.append(n_step_target)
                                n_step_next_ob = n_step_next_obs[n - 1 + n_steps]
                                n_step_done = team_n_step_dones[n - 1 + n_steps]
                            else: # n-step = MC if less than n steps remaining
                                n_step_target = MC_target
                                n_step_targets.append(n_step_target)
                                n_step_next_ob = n_step_next_obs[-1]
                                n_step_done = team_n_step_dones[-1]
                        # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                        #pretrain_buffer.push
                        #team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                        #                        [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                        #                        [team_n_step_ws[n] for i in range(env.num_TA)])
                        team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                                                [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                                                [team_n_step_ws[n] for i in range(env.num_TA)])

                    pt_time_step +=1
                    break
        
        del team_pt_obs
        del team_pt_status
        del team_pt_actions
        ################## Base Right ########################

        pt_time_step = 0
        for ep_i in range(0, pt_episodes):
            if ep_i % 100 == 0:
                print("Pushing Pretrain Base Right Episode:",ep_i)
                
                
            # team n-step
            team_n_step_rewards = []
            team_n_step_obs = []
            team_n_step_acs = []
            n_step_next_obs = []
            team_n_step_dones = []
            team_n_step_ws = []

            #define/update the noise used for exploration
            explr_pct_remaining = 0.0
            beta_pct_remaining = 0.0
            maddpg.scale_noise(0.0)
            maddpg.reset_noise()
            maddpg.scale_beta(pt_beta)
            d = False
            
            for et_i in range(0, episode_length):            
                world_stat = opp_pt_status[pt_time_step]
                d = False
                if world_stat != 0.0:
                    d = True

                #### Team ####
                team_n_step_acs.append(opp_pt_actions[pt_time_step])
                team_n_step_obs.append(np.array(opp_pt_obs[pt_time_step]).T)
                team_n_step_ws.append(world_stat)
                n_step_next_obs.append(np.array(opp_pt_obs[pt_time_step+1]).T)
                team_n_step_rewards.append(np.hstack([env.getPretrainRew(world_stat,d,"base_left") for i in range(env.num_TA) ]))          
                team_n_step_dones.append(d)

                # Store variables for calculation of MC and n-step targets for team
                pt_time_step += 1
                if d == True: # Episode done
                    # Calculate n-step and MC targets
                    for n in range(et_i+1):
                        MC_targets = []
                        n_step_targets = []
                        for a in range(env.num_TA):
                            MC_target = 0
                            n_step_target = 0
                            
                            for step in range(et_i+1 - n): # sum MC target
                                MC_target += team_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
                            MC_targets.append(MC_target)
                            if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                                for step in range(n_steps): 
                                    n_step_target += team_n_step_rewards[n + step][a] * gamma**(step)
                                n_step_targets.append(n_step_target)
                                n_step_next_ob = n_step_next_obs[n - 1 + n_steps]
                                n_step_done = team_n_step_dones[n - 1 + n_steps]
                            else: # n-step = MC if less than n steps remaining
                                n_step_target = MC_target
                                n_step_targets.append(n_step_target)
                                n_step_next_ob = n_step_next_obs[-1]
                                n_step_done = team_n_step_dones[-1]
                        # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                        #pretrain_buffer.push
                        #team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                        #                        [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                        #                        [team_n_step_ws[n] for i in range(env.num_TA)])
                        team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                                                [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                                                [team_n_step_ws[n] for i in range(env.num_TA)])

                    pt_time_step +=1
                    break
                    

        del opp_pt_obs
        del opp_pt_status
        del opp_pt_actions
                    ##################################################

        maddpg.prep_training(device=device)
        
        if I2A:
            # pretrain EM
            for i in range(pt_EM_updates):
                if i%100 == 0:
                    print("Petrain EM update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        #sample = pretrain_buffer.sample(batch_size,
                        sample = team_replay_buffer.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                        maddpg.update_EM(sample, a_i,'team')
                    maddpg.niter+=1
                        
            if Imitation_exploration:
                # pretrain policy prime
                    # pretrain critic
                for i in range(pt_actor_updates):
                    if i%100 == 0:
                        print("Petrain Prime update:",i)
                    for u_i in range(1):
                        for a_i in range(env.num_TA):
                            inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                            sample = team_replay_buffer.sample(inds,
                                                            to_gpu=to_gpu,norm_rews=False)
                            maddpg.pretrain_prime(sample, a_i,'team')
                        maddpg.niter +=1
                if imagination_policy_branch and I2A:
                    for i in range(pt_imagination_branch_pol_updates):
                        if i%100 == 0:
                            print("Petrain imag policy update:",i)
                        for u_i in range(1):
                            for a_i in range(env.num_TA):
                                inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                                sample = team_replay_buffer.sample(inds,
                                                            to_gpu=to_gpu,norm_rews=False)
                                maddpg.pretrain_imagination_policy(sample, a_i,'team')
                            maddpg.niter +=1

                    

        # pretrain policy
        for i in range(pt_actor_updates):
            if i%100 == 0:
                print("Petrain actor update:",i)
            for u_i in range(1):
                for a_i in range(env.num_TA):
                    inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                    sample = team_replay_buffer.sample(inds,
                                                to_gpu=to_gpu,norm_rews=False)
                    maddpg.pretrain_actor(sample, a_i,'team')
                maddpg.niter +=1

        maddpg.update_hard_policy()
        
        
        
        if not critic_mod: # non-centralized Q
            # pretrain critic
            for i in range(pt_critic_updates):
                if i%100 == 0:
                    print("Petrain critic update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        sample = team_replay_buffer.sample(inds,
                                                        to_gpu=to_gpu,norm_rews=False)
                        maddpg.pretrain_critic(sample, a_i,'team')
                    maddpg.niter +=1
            maddpg.update_hard_critic()


            maddpg.scale_beta(initial_beta) # 
            # pretrain true actor-critic (non-imitation) + policy prime
            for i in range(pt_actor_critic_updates):
                if i%100 == 0:
                    print("Petrain critic/actor update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        sample = team_replay_buffer.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                        maddpg.update(sample, a_i, 'team' )
                        if SIL:
                            for i in range(SIL_update_ratio):
                                team_sample,inds = team_replay_buffer.sample_SIL(agentID=a_i,batch_size=batch_size,
                                                    to_gpu=to_gpu,norm_rews=False)
                                priorities = maddpg.SIL_update(team_sample, opp_sample, a_i, 'team', 
                                                centQ=critic_mod) # 
                                team_replay_buffer.update_priorities(agentID=a_i,inds = inds, prio=priorities)
                    maddpg.update_all_targets()
        else: # centralized Q
            for i in range(pt_critic_updates):
                if i%100 == 0:
                    print("Petrain critic update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        team_sample = team_replay_buffer.sample(inds,
                                                                to_gpu=to_gpu,norm_rews=False)
                        opp_sample = opp_replay_buffer.sample(inds,
                                                            to_gpu=to_gpu,norm_rews=False)
                        maddpg.pretrain_centralized_critic(team_sample, opp_sample, a_i, 'team', 
                                                        act_only=critic_mod_act, obs_only=critic_mod_obs)
                    maddpg.niter +=1
            maddpg.update_hard_critic()
            
            maddpg.scale_beta(initial_beta) # 
            # pretrain true actor-critic update
            for i in range(pt_actor_critic_updates):
                if i%100 == 0:
                    print("Petrain critic/actor update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        team_sample = team_replay_buffer.sample(inds,
                                                                to_gpu=to_gpu,norm_rews=False)
                        opp_sample = opp_replay_buffer.sample(inds,
                                                            to_gpu=to_gpu,norm_rews=False)
                        maddpg.update_centralized_critic(team_sample, opp_sample, a_i, 'team', 
                                                        act_only=critic_mod_act, obs_only=critic_mod_obs)
                        if SIL:
                            for i in range(SIL_update_ratio):
                                team_sample,inds = team_replay_buffer.sample_SIL(agentID=a_i,batch_size=batch_size,
                                                    to_gpu=to_gpu,norm_rews=False)
                                opp_sample = opp_replay_buffer.sample(inds,to_gpu=to_gpu,norm_rews=False)
                                priorities = maddpg.SIL_update(team_sample, opp_sample, a_i, 'team', 
                                                centQ=critic_mod) # 
                                team_replay_buffer.update_priorities(agentID=a_i,inds = inds, prio=priorities)
                    maddpg.update_all_targets()
            
            

        

    # END PRETRAIN ###################
    # --------------------------------
    env.launch()
    if use_viewer:
        env._start_viewer()       

    time.sleep(3)
    for ep_i in range(0, num_episodes):
        start = time.time()
        # team n-step
        team_n_step_rewards = []
        team_n_step_obs = []
        team_n_step_acs = []
        team_n_step_next_obs = []
        team_n_step_dones = []
        team_n_step_ws = []


        # opp n-step
        opp_n_step_rewards = []
        opp_n_step_obs = []
        opp_n_step_acs = []
        opp_n_step_next_obs = []
        opp_n_step_dones = []
        opp_n_step_ws = []
        #maddpg.prep_rollouts(device='cpu')


        
        #define/update the noise used for exploration
        if ep_i < burn_in_episodes:
            explr_pct_remaining = 1.0
        else:
            explr_pct_remaining = max(0.0, 1.0*num_explore_episodes - ep_i + burn_in_episodes) / (num_explore_episodes)
        beta_pct_remaining = max(0.0, 1.0*num_beta_episodes - ep_i + burn_in_episodes) / (num_beta_episodes)
        
        # evaluation for 10 episodes every 100
        if ep_i % 10 == 0:
            maddpg.scale_noise(final_OU_noise_scale + (init_noise_scale - final_OU_noise_scale) * explr_pct_remaining)
        if ep_i % 100 == 0:
            maddpg.scale_noise(0.0)

        if LSTM or LSTM_PC:
            maddpg.reset_hidden(training=False)
        maddpg.reset_noise()
        maddpg.scale_beta(final_beta + (initial_beta - final_beta) * beta_pct_remaining)
        #for the duration of 100 episode with maximum length of 500 time steps
        time_step = 0
        team_kickable_counter = [0] * num_TA
        opp_kickable_counter = [0] * num_OA
        env.team_possession_counter = [0] * num_TA
        env.opp_possession_counter = [0] * num_OA

        for et_i in range(0, episode_length):
            maddpg.prep_training(device=device) # GPU for forward passes?? 

            # gather all the observations into a torch tensor 
            torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False).cuda(non_blocking=True)
                        for i in range(maddpg.nagents_team)]

            # gather all the opponent observations into a torch tensor 
            torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False).cuda(non_blocking=True)
                        for i in range(maddpg.nagents_opp)]

            # Get e-greedy decision
            if explore:
                team_randoms = e_greedy_bool(env.num_TA,eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining),device=device)
                opp_randoms = e_greedy_bool(env.num_OA,eps = 0,device=device)
            else:
                team_randoms = e_greedy_bool(env.num_TA,eps = 0,device=device)
                opp_randoms = e_greedy_bool(env.num_OA,eps = 0,device=device)

            # get actions as torch Variables for both team and opp

            # change parallel to parallel_process when fixed **
            team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_opp,team_randoms,opp_randoms,parallel=False,explore=explore) # leave off or will gumbel sample
            # convert actions to numpy arrays

            team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
            #Converting actions to numpy arrays for opp agents
            opp_agent_actions = [ac.cpu().data.numpy() for ac in opp_torch_agent_actions]

            # rearrange actions to be per environment
            team_params = np.asarray([ac[0][len(env.action_list):] for ac in team_agent_actions]) 
            # rearrange actions to be per environment for the opponent
            opp_params = np.asarray([ac[0][len(env.action_list):] for ac in opp_agent_actions])

            # this is returning one-hot-encoded action for each team agent
            team_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in team_agent_actions]])
            # this is returning one-hot-encoded action for each opp agent 
            opp_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in opp_agent_actions]])

            
            #if ep_i % 10 == 0:
            #    explore = True
            #if ep_i % 100 == 0:
            #    explore = False

            team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            opp_obs =  np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T
            
            # use random unif parameters if e_greedy
            team_noisey_actions_for_buffer = team_actions[0]
            team_params = np.array([val[0][len(env.action_list):] for val in team_agent_actions])
            opp_noisey_actions_for_buffer = opp_actions[0]
            opp_params = np.array([val[0][len(env.action_list):] for val in opp_agent_actions])

            # print('These are the opp params', str(opp_params))

            team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes
            #Added to Disable/Enable the team agents
            if has_team_Agents:        
                team_actions_params_for_buffer = np.array([val[0] for val in team_agent_actions])
            opp_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in opp_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_actions_params_for_buffer = np.array([val[0] for val in opp_agent_actions])

            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.get_kickable_status(i,team_obs.T) for i in range(env.num_TA)])
            if kickable.any():
                team_kickable_counter = [tkc + 1 if kickable[i] else tkc for i,tkc in enumerate(team_kickable_counter)]
                
            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.get_kickable_status(i,opp_obs.T) for i in range(env.num_OA)])
            if kickable.any():
                opp_kickable_counter = [okc + 1 if kickable[i] else okc for i,okc in enumerate(opp_kickable_counter)]
            
            team_possession_counter = [env.get_agent_possession_status(i, env.team_base) for i in range(num_TA)]
            opp_possession_counter = [env.get_agent_possession_status(i, env.opp_base) for i in range(num_OA)]

            _,_,_,_,d,world_stat = env.Step(team_agents_actions, opp_agents_actions, team_params, opp_params)

            #Added to Disable/Enable the team agents
            if has_team_Agents:
                team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_rewards = np.hstack([env.Reward(i,'opp') for i in range(env.num_OA)])

            #Added to Disable/Enable the team agents
            if has_team_Agents:
                team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_next_obs = np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T

            
            team_done = env.d
            opp_done = env.d 

            # Store variables for calculation of MC and n-step targets for team
            #Added to Disable/Enable the team agents
            if has_team_Agents:
                team_n_step_rewards.append(team_rewards)
                team_n_step_obs.append(team_obs)
                team_n_step_next_obs.append(team_next_obs)
                team_n_step_acs.append(team_actions_params_for_buffer)
                team_n_step_dones.append(team_done)
                team_n_step_ws.append(world_stat)
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_n_step_rewards.append(opp_rewards)
                opp_n_step_obs.append(opp_obs)
                opp_n_step_next_obs.append(opp_next_obs)
                opp_n_step_acs.append(opp_actions_params_for_buffer)
                opp_n_step_dones.append(opp_done)
                opp_n_step_ws.append(world_stat)
            # ----------------------------------------------------------------

            training = ((len(ptr_buff) >= batch_size and (t % steps_per_update) < 1) and t > burn_in_iterations)

            if training:
                #print('****************We are now training*********************')
                maddpg.prep_training(device=device)
                if not critic_mod_both:
                    for u_i in range(1):
                        if train_team: # train team
                            for _ in range(number_of_updates):
                                for a_i in range(maddpg.nagents_team):
                                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                                        sample = team_replay_buffer.sample(inds,
                                                                    to_gpu=to_gpu,norm_rews=False)
                                        maddpg.update(sample, a_i, 'team')
                                        if SIL:
                                            for i in range(SIL_update_ratio):
                                                team_sample,inds = team_replay_buffer.sample_SIL(agentID=a_i,batch_size=batch_size,
                                                                    to_gpu=to_gpu,norm_rews=False)
                                                priorities = maddpg.SIL_update(team_sample=team_sample,agent_i= a_i, side='team', 
                                                                centQ=critic_mod_both) #
                                                team_replay_buffer.update_priorities(agentID=a_i,inds = inds, prio=priorities)
                                
                                maddpg.update_all_targets()
                    # maddpg.prep_rollouts(device='cpu') convert back to cpu for pushing?
                else:
                    for u_i in range(1):
                        # NOTE: Only works for m vs m
                        if train_team: # train team      
                            if parallel_process:
                                maddpg_pick = dill.dumps(maddpg)
                                threads = []
                                for a_i in range(maddpg.nagents_team):
                                    threads.append(mp.Process(target=update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                                        team_replay_buffer,opp_replay_buffer,critic_mod_act,critic_mod_obs,number_of_updates,maddpg_pick,
                                        load_path,ep_i,ensemble_path,current_ensembles,forward_pass)))
                                print("Starting threads")
                                start = time.time()                                    

                                [thr.start() for thr in threads]
                                print(time.time()-start)
                            else:
                                start = time.time()
                                for _ in range(number_of_updates):
                                    for a_i in range(maddpg.nagents_team):                                            
                                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)

                                        if LSTM:
                                            team_sample = team_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                                            opp_sample = opp_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                                            maddpg.update_centralized_critic_LSTM(team_sample, opp_sample, a_i, 'team',     
                                                                            act_only=critic_mod_act, obs_only=critic_mod_obs)
                                        elif LSTM_PC:
                                            team_sample = team_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                                            opp_sample = opp_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                                            maddpg.update_centralized_critic_LSTM_PC(team_sample, opp_sample, a_i, 'team', 
                                                                            act_only=critic_mod_act, obs_only=critic_mod_obs)
                                        else:
                                            team_sample = team_replay_buffer.sample(inds,
                                                                        to_gpu=to_gpu,norm_rews=False)
                                            opp_sample = opp_replay_buffer.sample(inds,
                                                                        to_gpu=to_gpu,norm_rews=False)


                                            maddpg.update_centralized_critic(team_sample, opp_sample, a_i, 'team', 
                                                                            act_only=critic_mod_act, obs_only=critic_mod_obs)

                                        if SIL:
                                            for i in range(SIL_update_ratio):
                                                team_sample,inds = team_replay_buffer.sample_SIL(agentID=a_i,batch_size=batch_size,
                                                                    to_gpu=to_gpu,norm_rews=False)
                                                opp_sample = opp_replay_buffer.sample(inds,to_gpu=to_gpu,norm_rews=False)
                                                priorities = maddpg.SIL_update(team_sample, opp_sample, a_i, 'team', 
                                                                centQ=critic_mod_both) # 
                                                team_replay_buffer.update_priorities(agentID=a_i,inds = inds, prio=priorities)
                                        maddpg.update_all_targets()
                                print(time.time()-start)


            time_step += 1
            t += 1

            if t%10000 == 0:
                team_step_logger_df.to_csv(hist_dir + '/team_%s.csv' % history)
                opp_step_logger_df.to_csv(hist_dir + '/opp_%s.csv' % history)
                        
        
                        
            if d == True and et_i >= (trace_length-1): # Episode done 
                # ----------- Push Base Left's experiences to team buffer ---------------------    

                team_all_MC_targets = []
                # calculate MC
                MC_targets = np.zeros(env.num_TA)
                for n in range(et_i +1):
                    MC_targets = team_n_step_rewards[et_i-n] + MC_targets*gamma
                    team_all_MC_targets.append(MC_targets)
                for n in range(et_i+1):
                    n_step_targets = np.zeros(env.num_TA)
                    if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                        for step in range(n_steps): 
                            n_step_targets += team_n_step_rewards[n + step] * gamma**(step)
                        n_step_next_ob = team_n_step_next_obs[n - 1 + n_steps]
                        n_step_done = team_n_step_dones[n - 1 + n_steps]
                    else: # n-step = MC if less than n steps remaining
                        n_step_targets = team_all_MC_targets[et_i-n]
                        n_step_next_ob = team_n_step_next_obs[-1]
                        n_step_done = team_n_step_dones[-1] 
                    # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                    if LSTM or LSTM_PC:
                        if n == et_i:
                            team_replay_buffer.done_step = True
                        team_replay_buffer.push_LSTM(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],
                                                n_step_next_ob,[n_step_done for i in range(env.num_TA)],team_all_MC_targets[et_i-n],
                                                n_step_targets,[team_n_step_ws[n] for i in range(env.num_TA)])
                    else:
                        team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],
                                                n_step_next_ob,[n_step_done for i in range(env.num_TA)],team_all_MC_targets[et_i-n],
                                                n_step_targets,[team_n_step_ws[n] for i in range(env.num_TA)])

                # -------------------- Push Base Right's experiences opp buffer and team buffer --------------------------
                if train_opp or critic_mod_both: # only create experiences if critic mod or training opp
                    opp_all_MC_targets = []
                    MC_targets = np.zeros(env.num_OA)
                    # calculate MC
                    for n in range(et_i +1):
                        MC_targets = opp_n_step_rewards[et_i-n] + MC_targets*gamma
                        opp_all_MC_targets.append(MC_targets)                    
                    for n in range(et_i+1):
                        n_step_targets = np.zeros(env.num_OA)
                        if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                            for step in range(n_steps): 
                                n_step_targets += opp_n_step_rewards[n + step] * gamma**(step)
                            n_step_next_ob = opp_n_step_next_obs[n - 1 + n_steps]
                            n_step_done = opp_n_step_dones[n - 1 + n_steps]
                        else: # n-step = MC if less than n steps remaining
                            n_step_targets = opp_all_MC_targets[et_i-n]
                            n_step_next_ob = opp_n_step_next_obs[-1]
                            n_step_done = opp_n_step_dones[-1] 
                            # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                        if LSTM or LSTM_PC:
                            if n == et_i:
                                team_replay_buffer.done_step = True
                            team_replay_buffer.push_LSTM(opp_n_step_obs[n], opp_n_step_acs[n],opp_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_OA)],opp_all_MC_targets[et_i-n],
                                                    n_step_targets,[opp_n_step_ws[n] for i in range(env.num_OA)])
                        else:
                            team_replay_buffer.push(opp_n_step_obs[n], opp_n_step_acs[n],opp_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_OA)],opp_all_MC_targets[et_i-n],
                                                    n_step_targets,[opp_n_step_ws[n] for i in range(env.num_OA)])
                        
                        if has_opp_Agents:
                            if LSTM or LSTM_PC:
                                if n == et_i:
                                    opp_replay_buffer.done_step = True
                                opp_replay_buffer.push_LSTM(opp_n_step_obs[n], opp_n_step_acs[n],opp_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_OA)],opp_all_MC_targets[et_i-n],
                                                    n_step_targets,[opp_n_step_ws[n] for i in range(env.num_OA)])
                            else:
                                opp_replay_buffer.push(opp_n_step_obs[n], opp_n_step_acs[n],opp_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_OA)],opp_all_MC_targets[et_i-n],
                                                    n_step_targets,[opp_n_step_ws[n] for i in range(env.num_OA)])
                    # ------------------- Push Base Left's Experiences to opp buffer to balance CentQ ------------------
                    ###################### Update this so that the n_step_targets and n_step_ob are in arrays so that we do not have to recalculate anything #####################
                    for n in range(et_i+1):
                        n_step_targets = np.zeros(env.num_OA)
                        if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                            for step in range(n_steps): 
                                n_step_targets += opp_n_step_rewards[n + step] * gamma**(step)
                            n_step_next_ob = opp_n_step_next_obs[n - 1 + n_steps]
                            n_step_done = opp_n_step_dones[n - 1 + n_steps]
                        else: # n-step = MC if less than n steps remaining
                            n_step_targets = opp_all_MC_targets[et_i-n]
                            n_step_next_ob = opp_n_step_next_obs[-1]
                            n_step_done = opp_n_step_dones[-1] 
                            # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                        if LSTM or LSTM_PC:
                            if n == et_i:
                                opp_replay_buffer.done_step = True
                            opp_replay_buffer.push_LSTM(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_TA)],team_all_MC_targets[et_i-n],
                                                    n_step_targets,[team_n_step_ws[n] for i in range(env.num_TA)])
                        else:
                            opp_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],
                                                    n_step_next_ob,[n_step_done for i in range(env.num_TA)],team_all_MC_targets[et_i-n],
                                                    n_step_targets,[team_n_step_ws[n] for i in range(env.num_TA)])
                    #############################################################################################################################################################

                # log
                if ep_i > 1:
                    #Added to Disable/Enable the team agents
                    if has_team_Agents:
                        team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(tkc/time_step)*100 for tkc in team_kickable_counter],
                                                            'possession_percentages': [(tpc/time_step)*100 for tpc in team_possession_counter],
                                                            'average_reward': team_replay_buffer.get_average_rewards(time_step),
                                                            'cumulative_reward': team_replay_buffer.get_cumulative_rewards(time_step)},
                                                            ignore_index=True)

                    #Added to Disable/Enable the opp agents
                    if has_opp_Agents:
                        opp_step_logger_df = opp_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.opp_team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(okc/time_step)*100 for okc in opp_kickable_counter],
                                                            'possession_percentages': [(opc/time_step)*100 for opc in opp_possession_counter],
                                                            'average_reward': opp_replay_buffer.get_average_rewards(time_step),
                                                            'cumulative_reward': opp_replay_buffer.get_cumulative_rewards(time_step)},
                                                            ignore_index=True)

                if first_save: # Generate list of ensemble networks
                    file_path = ensemble_path
                    maddpg.first_save(file_path,num_copies = k_ensembles)
                    first_save = False
                # Save networks
                if not parallel_process: # if parallel saving occurs in update thread
                        
                    if ep_i > 1 and ep_i%ep_save_every == 0 and save_nns:
                        maddpg.save(load_path,ep_i)
                        maddpg.save_ensembles(ensemble_path,current_ensembles)

                # Launch evaluation session
                if ep_i > 1 and ep_i % eval_after == 0 and evaluate:
                    thread.start_new_thread(launch_eval,(
                        [load_path + ("agent_%i/model_episode_%i.pth" % (i,ep_i)) for i in range(env.num_TA)], # models directory -> agent -> most current episode
                        eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation_ep" + str(ep_i),
                        7000,env.num_TA,env.num_OA,episode_length,device,use_viewer,))

                # Load random networks into team from ensemble and opponent from all models
                if ep_i > ep_save_every and ep_i % load_random_every == 0 and load_random_nets:
                    if parallel_process:
                        [thr.join() for thr in threads] # join here before loading net

                    if np.random.uniform(0,1) > self_play_proba: # self_play_proba % chance loading self else load an old ensemble for opponent
                        maddpg.load_random_policy(side='opp',nagents = num_OA,models_path = load_path)
                    else:
                        maddpg.load_random_ensemble(side='opp',nagents = num_OA,models_path = ensemble_path)
                    current_ensembles = maddpg.load_random_ensemble(side='team',nagents=num_TA,models_path = ensemble_path)

                end = time.time()
                print(end-start)

                break
            elif d:
                break

                    
                
            team_obs = team_next_obs
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_obs = opp_next_obs

                #print(step_logger_df) 
            #if t%30000 == 0 and use_viewer:
            if t%20000 == 0 and use_viewer and ep_i > use_viewer_after:
                env._start_viewer()       


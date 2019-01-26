import re
import itertools
import random
import datetime
import os 
import csv
import itertools 
import argparse
import pandas as pd
#import tensorflow.contrib.slim as slim
import numpy as np
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits,e_greedy,zero_params,pretrain_process,prep_session,e_greedy_bool
from HFO import hfo
import time
import _thread as thread
import torch
from pathlib import Path
from torch.autograd import Variable
# from tensorboardX import SummaryWriter
from utils.tensor_buffer import ReplayTensorBuffer
from algorithms.maddpg import MADDPG
from HFO_env import HFO_env
from trainer import launch_eval
import torch.multiprocessing as mp
import _thread as thread

def run_envs(seed, port, se, sfs, exp_i, env_id):
    
    # Parseargs --------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Load port and log directory')
    parser.add_argument('-log_dir', type=str, default='log',
                    help='A name for log directory')
    parser.add_argument('-log', type=str, default='history',
                    help='A name for log file ')

    args = parser.parse_args()
    log_dir = args.log_dir
    history = args.log
    # --------------------------------------

    # options ------------------------------
    action_level = 'low'
    feature_level = 'low'
    USE_CUDA = False
    if USE_CUDA:
        device = 'cuda'
        to_gpu = True
    else:
        to_gpu = False
        device = 'cpu'

    use_viewer = False
    use_viewer_after = 1000 # If using viewer, uses after x episodes
    n_training_threads = 4
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
    num_TA = 1
    num_OA = 1
    num_TNPC = 0
    num_ONPC = 0
    offense_team_bin='helios10'
    defense_team_bin='helios11'  
    goalie = False
    team_rew_anneal_ep = 1500 # reward would be
    # hyperparams--------------------------
    batch_size = 512
    hidden_dim = int(512)
    a_lr = 0.0001 # actor learning rate
    c_lr = 0.001 # critic learning rate
    tau = 0.001 # soft update rate
    steps_per_update = 5000
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
    # session_path = None
    # current_day_time = datetime.datetime.now()
    # session_path = 'training_sessions/' + \
    #                                 str(current_day_time.month) + \
    #                                 '_' + str(current_day_time.day) + \
    #                                 '_' + str(current_day_time.hour) + '_' + \
    #                                 str(num_TA) + '_vs_' + str(num_OA) + "/"
    # hist_dir = session_path +"history"
    # eval_hist_dir = session_path +"eval_history"
    # eval_log_dir = session_path +"eval_log" # evaluation logfiles
    # load_path = session_path +"models/"
    # ensemble_path = session_path +"ensemble_models/"
    # prep_session(session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,log_dir,num_TA) # Generates directories and files for the session

    # --------------------------------------
    # if using low level actions use non discrete settings
    if action_level == 'high':
        discrete_action = True
    else:
        discrete_action = False
    if not USE_CUDA:
        torch.set_num_threads(n_training_threads)

    #Initialization for either M vs M or M vs 0
    if num_OA > 0:
        has_opp_Agents = True
    else:
        has_opp_Agents = False

    env = HFO_env(num_TNPC = num_TNPC,num_TA=num_TA,num_OA=num_OA, num_ONPC=num_ONPC, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length,
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=port, seed=seed, 
                    log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game, team_rew_anneal_ep=team_rew_anneal_ep,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=True,
                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin)
    
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

    env.launch()
    if use_viewer:
        env._start_viewer()
    
    time.sleep(3)
    t = 0
    time_step = 0
    exps = None

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

            if device == 'cuda':
                # gather all the observations into a torch tensor 
                torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False).cuda(non_blocking=True)
                            for i in range(num_TA)]

                # gather all the opponent observations into a torch tensor 
                torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False).cuda(non_blocking=True)
                            for i in range(num_OA)]
            else:
                # gather all the observations into a torch tensor 
                torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False)
                            for i in range(num_TA)]

                # gather all the opponent observations into a torch tensor 
                torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False)
                            for i in range(num_OA)]

            # Get e-greedy decision
            if explore:
                team_randoms = e_greedy_bool(num_TA,eps = (final_noise_scale + (init_noise_scale - final_noise_scale) * explr_pct_remaining),device=device)
                opp_randoms = e_greedy_bool(num_OA,eps = 0,device=device)
            else:
                team_randoms = e_greedy_bool(num_TA,eps = 0,device=device)
                opp_randoms = e_greedy_bool(num_OA,eps = 0,device=device)

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

            team_obs =  np.array([env.Observation(i,'team') for i in range(num_TA)]).T
            opp_obs =  np.array([env.Observation(i,'opp') for i in range(num_OA)]).T
            
            # use random unif parameters if e_greedy
            team_noisey_actions_for_buffer = team_actions[0]
            team_params = np.array([val[0][len(env.action_list):] for val in team_agent_actions])
            opp_noisey_actions_for_buffer = opp_actions[0]
            opp_params = np.array([val[0][len(env.action_list):] for val in opp_agent_actions])

            # print('These are the opp params', str(opp_params))

            team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes       
            team_actions_params_for_buffer = np.array([val[0] for val in team_agent_actions])
            opp_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in opp_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_actions_params_for_buffer = np.array([val[0] for val in opp_agent_actions])

            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.get_kickable_status(i,team_obs.T) for i in range(num_TA)])
            if kickable.any():
                team_kickable_counter = [tkc + 1 if kickable[i] else tkc for i,tkc in enumerate(team_kickable_counter)]
                
            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.get_kickable_status(i,opp_obs.T) for i in range(num_OA)])
            if kickable.any():
                opp_kickable_counter = [okc + 1 if kickable[i] else okc for i,okc in enumerate(opp_kickable_counter)]
            
            team_possession_counter = [env.get_agent_possession_status(i, env.team_base) for i in range(num_TA)]
            opp_possession_counter = [env.get_agent_possession_status(i, env.opp_base) for i in range(num_OA)]

            _,_,_,_,d,world_stat = env.Step(team_agents_actions, opp_agents_actions, team_params, opp_params)

            team_rewards = np.hstack([env.Reward(i,'team') for i in range(num_TA)])
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_rewards = np.hstack([env.Reward(i,'opp') for i in range(num_OA)])

            team_next_obs = np.array([env.Observation(i,'team') for i in range(num_TA)]).T
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_next_obs = np.array([env.Observation(i,'opp') for i in range(num_OA)]).T

            # Store variables for calculation of MC and n-step targets for team
            team_n_step_rewards.append(team_rewards)
            team_n_step_obs.append(team_obs)
            team_n_step_next_obs.append(team_next_obs)
            team_n_step_acs.append(team_actions_params_for_buffer)
            team_n_step_dones.append(d)
            team_n_step_ws.append(world_stat)
            #Added to Disable/Enable the opp agents
            if has_opp_Agents:
                opp_n_step_rewards.append(opp_rewards)
                opp_n_step_obs.append(opp_obs)
                opp_n_step_next_obs.append(opp_next_obs)
                opp_n_step_acs.append(opp_actions_params_for_buffer)
                opp_n_step_dones.append(d)
                opp_n_step_ws.append(world_stat)
            # ----------------------------------------------------------------

            time_step += 1
            t += 1

            # if t%10000 == 0:
            #     team_step_logger_df.to_csv(hist_dir + '/team_%s.csv' % history)
            #     opp_step_logger_df.to_csv(hist_dir + '/opp_%s.csv' % history)       
        
            #start = time.time()
            if d == True and et_i >= (trace_length-1): # Episode done

                #NOTE: Assume M vs M and critic_mod_both == True
                if critic_mod_both:
                    team_all_MC_targets = []
                    opp_all_MC_targets = []
                    MC_targets_team = np.zeros(num_TA)
                    MC_targets_opp = np.zeros(num_OA)
                    for n in range(et_i+1):
                        MC_targets_team = team_n_step_rewards[et_i-n] + MC_targets_team*gamma
                        team_all_MC_targets.append(MC_targets_team)
                        MC_targets_opp = opp_n_step_rewards[et_i-n] + MC_targets_opp*gamma
                        opp_all_MC_targets.append(MC_targets_opp)
                    for n in range(et_i+1):
                        n_step_targets_team = np.zeros(num_TA)
                        n_step_targets_opp = np.zeros(num_OA)
                        if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                            for step in range(n_steps):
                                n_step_targets_team += team_n_step_rewards[n + step] * gamma**(step)
                                n_step_targets_opp += opp_n_step_rewards[n + step] * gamma**(step)
                            n_step_next_ob_team = team_n_step_next_obs[n - 1 + n_steps]
                            n_step_done_team = team_n_step_dones[n - 1 + n_steps]

                            n_step_next_ob_opp = opp_n_step_next_obs[n - 1 + n_steps]
                            n_step_done_opp = opp_n_step_dones[n - 1 + n_steps]
                        else: # n-step = MC if less than n steps remaining
                            n_step_targets_team = team_all_MC_targets[et_i-n]
                            n_step_next_ob_team = team_n_step_next_obs[-1]
                            n_step_done_team = team_n_step_dones[-1]

                            n_step_targets_opp = opp_all_MC_targets[et_i-n]
                            n_step_next_ob_opp = opp_n_step_next_obs[-1]
                            n_step_done_opp = opp_n_step_dones[-1]
                        
                        exp_team = np.column_stack((np.transpose(team_n_step_obs[n]),
                                            team_n_step_acs[n],
                                            np.expand_dims(team_n_step_rewards[n], 1),
                                            np.transpose(n_step_next_ob_team),
                                            np.expand_dims([n_step_done_team for i in range(num_TA)], 1),
                                            np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_team, 1),
                                            np.expand_dims([team_n_step_ws[n] for i in range(num_TA)], 1)))

                        exp_opp = np.column_stack((np.transpose(opp_n_step_obs[n]),
                                            opp_n_step_acs[n],
                                            np.expand_dims(opp_n_step_rewards[n], 1),
                                            np.transpose(n_step_next_ob_opp),
                                            np.expand_dims([n_step_done_opp for i in range(num_OA)], 1),
                                            np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_opp, 1),
                                            np.expand_dims([opp_n_step_ws[n] for i in range(num_OA)], 1)))
                        
                        exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)

                        if exps is None:
                            exps = torch.from_numpy(exp_comb)
                        else:
                            exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)

                    exp_i += et_i
                    if (ep_i+1) % 5 == 0:
                        # print('exps', exps[:len(exps)][0][0])
                        se[:len(exps)] = exps
                        sfs[env_id] = 1
                        while sfs[env_id] == 1:
                            time.sleep(0.0000001)
                        exp_i = 0

                        # print('The length', len(exps))
                        # print(se)

                    print('torch write')

                #############################################################################################################################################################

                print(time.time()-start)
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

def read(ses, sfs, exps_i):

    num_TA = 1
    num_OA = 1
    replay_memory_size = 100000
    obs_dim_TA = 68+(18*(num_TA-1))
    obs_dim_OA = 68+(18*(num_OA-1))
    acs_dim = 8
    batch_size = 512
    LSTM = False
    LSTM_PC = False

    team_buffer = ReplayTensorBuffer(replay_memory_size, num_TA, obs_dim_TA, acs_dim, batch_size, LSTM, LSTM_PC)
    opp_buffer = ReplayTensorBuffer(replay_memory_size, num_OA, obs_dim_OA, acs_dim, batch_size, LSTM, LSTM_PC)

    while(1):
        if sfs.byte().all():
            start = time.time()
            for i in range(num_TA):
                # print('ses', ses[i][:exps_i[i]][0][0])
                # print('exps_i', exps_i[i])
                print('entering replay')
                team_buffer.push(ses[i][:exps_i[i], :1, :])
                sample = team_buffer.sample(to_gpu=False, norm_rews=False)
                print('Done sampling', sample)
                exit(0)
                # team_buffer.push(ses[i][:exps_i[i], 1, :])

            sfs[:len(sfs)] = 0
            print('shared flag', sfs)
            print('read', (time.time()-start))


        start = time.time()
        # print('torch read', time.time()-start)



if __name__ == "__main__":
    mp.set_start_method('forkserver',force=True)

    # Multi envs --------------------------------------
    num_envs = 2
    seed = 123
    port = 6000
    # -------------------------------------------------

    # reward_total = []
    # num_steps_per_episode = []
    # end_actions = [] 
    # logger_df = pd.DataFrame()
    # team_step_logger_df = pd.DataFrame()
    # opp_step_logger_df = pd.DataFrame()
    # --------------------------------
    shared_exps = [torch.zeros(500*40, 2, 149) for _ in range(num_envs)]
    exp_indeces = [torch.tensor(0) for _ in range(num_envs)]
    share_flags = torch.zeros(num_envs)
    share_flags.share_memory_()
    processes = []

    for i in range(num_envs):
        shared_exps[i].share_memory_()
        exp_indeces[i].share_memory_()
        processes.append(mp.Process(target=run_envs, args=(seed + (i * 100), port + (i * 1000), shared_exps[i], share_flags, exp_indeces[i], i)))
    
    read_proc = mp.Process(target=read, args=(shared_exps, share_flags, exp_indeces))
    for p in processes:
        p.start()
    read_proc.start()
    for p in processes:
        p.join()
    read_proc.join()


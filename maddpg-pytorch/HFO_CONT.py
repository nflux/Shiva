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
from utils.tensor_buffer import ReplayTensorBuffer

#from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from HFO_env import *
from trainer import launch_eval
import torch.multiprocessing as mp
import _thread as thread
import dill



def update_thread(agentID,to_gpu,buffer_size,batch_size,team_replay_buffer,opp_replay_buffer,number_of_updates,
                            load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_PC,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu):
    start = time.time()
    initial_models = [ensemble_path + ("ensemble_agent_%i/model_%i.pth" % (i,0)) for i in range(num_TA)]
    #maddpg = dill.loads(maddpg_pick)
    maddpg = MADDPG.init_from_save_evaluation(initial_models,num_TA) # from evaluation method just loads the networks
    
    number_of_updates = 900
    for ensemble in range(k_ensembles):
        if multi_gpu:
            maddpg.torch_device = torch.device("cuda:1")
        maddpg.device = 'cuda'
        maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
        maddpg.load_same_ensembles(ensemble_path,ensemble,maddpg.nagents_team,load_same_agent=load_same_agent)

        #start = time.time()
        #for up in range(int(np.floor(number_of_updates/k_ensembles))):
        for up in range(number_of_updates):

            m = np.random.randint(num_TA)

            if not load_same_agent:
                inds = team_replay_buffer.get_PER_inds(agentID,batch_size,ensemble)
            else:
                inds = team_replay_buffer.get_PER_inds(m,batch_size,ensemble)

            #inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)

            # FOR THE LOVE OF GOD DONT USE TORCH TO GET INDICES

            if LSTM:
                team_sample = team_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                opp_sample = opp_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                priorities =maddpg.update_centralized_critic_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',load_same_agent=load_same_agent)
                team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                del priorities
                del team_sample
                del opp_sample


                if not load_same_agent:
                    priorities = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent)
                    team_replay_buffer.update_priorities(agentID=agentID,inds = inds, prio=priorities,k = ensemble)


            elif LSTM_PC:
                team_sample = team_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                opp_sample = opp_replay_buffer.sample_LSTM(inds, trace_length,to_gpu=to_gpu,norm_rews=False)
                maddpg.update_centralized_critic_LSTM_PC(team_sample, opp_sample, agentID, 'team')
            else:
                team_sample = team_replay_buffer.sample(inds,
                                            to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                opp_sample = opp_replay_buffer.sample(inds,
                                            to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                if not load_same_agent:
                    priorities = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent)
                    team_replay_buffer.update_priorities(agentID=agentID,inds = inds, prio=priorities,k = ensemble)

                else:
                    priorities = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent)
                    team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                    del priorities
                    del team_sample
                    del opp_sample
                    if up == number_of_updates/2: # update target half way through
                        maddpg.update_agent_targets(0,number_of_updates/2)





            if SIL:
                for i in range(SIL_update_ratio):
                    inds = team_replay_buffer.get_SIL_inds(agentID=m,batch_size=batch_size)
                    team_sample = team_replay_buffer.sample(inds,
                                            to_gpu=to_gpu,norm_rews=False)
                    opp_sample = opp_replay_buffer.sample(inds,
                                            to_gpu=to_gpu,norm_rews=False)
                    priorities = maddpg.SIL_update(team_sample, opp_sample, agentID, 'team') # 
                    team_replay_buffer.update_SIL_priorities(agentID=m,inds = inds, prio=priorities)
                    
        #print(time.time()-start)
        if not load_same_agent:
            maddpg.update_agent_targets(agentID,number_of_updates)
            maddpg.save_agent(load_path,update_session,agentID)
            maddpg.save_ensemble(ensemble_path,ensemble,agentID)
        else:
            maddpg.update_agent_targets(0,number_of_updates/2)
            [maddpg.save_agent(load_path,update_session,i,load_same_agent) for i in range(num_TA)]
            [maddpg.save_ensemble(ensemble_path,ensemble,i,load_same_agent) for i in range(num_TA)]
    print(time.time()-start,"<-- Update Cycle")


def run_envs(seed, port, shared_exps,exp_i,HP,env_num,ready,halt,num_updates,history,ep_num):

    (action_level,feature_level,to_gpu,device,use_viewer,use_viewer_after,n_training_threads,rcss_log_game,hfo_log_game,num_episodes,replay_memory_size,
    episode_length,untouched_time,burn_in_iterations,burn_in_episodes, deterministic, num_TA,num_OA,num_TNPC,num_ONPC,offense_team_bin,defense_team_bin,goalie,team_rew_anneal_ep,
    batch_size,hidden_dim,a_lr,c_lr,tau,steps_per_update,explore,final_OU_noise_scale,final_noise_scale,init_noise_scale,num_explore_episodes,D4PG,gamma,Vmax,Vmin,N_ATOMS,
    DELTA_Z,n_steps,initial_beta,final_beta,num_beta_episodes,TD3,TD3_delay_steps,TD3_noise,I2A,decent_EM,EM_lr,obs_weight,rew_weight,ws_weight,rollout_steps,
    LSTM_hidden,imagination_policy_branch,SIL,SIL_update_ratio,critic_mod_act,critic_mod_obs,critic_mod_both,control_rand_init,ball_x_min,ball_x_max,
    ball_y_min,ball_y_max,agents_x_min,agents_x_max,agents_y_min,agents_y_max,change_every_x,change_agents_x,change_agents_y,change_balls_x,change_balls_y,
    load_random_nets,load_random_every,k_ensembles,current_ensembles,self_play_proba,save_nns,load_nets,initial_models,evaluate,eval_after,eval_episodes,
    LSTM,LSTM_PC,trace_length,hidden_dim_lstm,parallel_process,forward_pass,session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,t,time_step,discrete_action,
    has_team_Agents,has_opp_Agents,log_dir,obs_dim_TA,obs_dim_OA, acs_dim,max_num_experiences,load_same_agent,multi_gpu) = HP


    env = HFO_env(num_TNPC = num_TNPC,num_TA=num_TA,num_OA=num_OA, num_ONPC=num_ONPC, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length, seed=seed, # create environment
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=port,log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game, team_rew_anneal_ep=team_rew_anneal_ep,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=True,
                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, run_server=True, deterministic=deterministic)

    #The start_viewer here is added to automatically start viewer with npc Vs npcs
    if num_TNPC > 0 and num_ONPC > 0:
        env._start_viewer() 

    time.sleep(3)
    print("Done connecting to the server ")

    # ------ Testing ---------
    #device = 'cpu'
    #to_gpu = False
    #cuda = False
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
                                LSTM=LSTM, LSTM_PC=LSTM_PC, trace_length=trace_length, hidden_dim_lstm=hidden_dim_lstm,only_policy=True,multi_gpu=multi_gpu) 
    maddpg.prep_training(device=maddpg.device,only_policy=True)

    reward_total = [ ]
    num_steps_per_episode = []
    end_actions = [] 
    # logger_df = pd.DataFrame()
    team_step_logger_df = pd.DataFrame()
    opp_step_logger_df = pd.DataFrame()

    exps = None

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

            time_step += 1
            t += 1

            if t%10000 == 0:
                team_step_logger_df.to_csv(hist_dir + '/team_%s.csv' % history)
                opp_step_logger_df.to_csv(hist_dir + '/opp_%s.csv' % history)
                        
            team_episode = []
            opp_episode = []

            if d == 1 and et_i >= (trace_length-1): # Episode done 
                n_step_gammas = np.array([[gamma**step for a in range(num_TA)] for step in range(n_steps)])
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
                            n_step_targets_team = np.sum(np.multiply(np.asarray(team_n_step_rewards[n:n+n_steps]),(n_step_gammas)),axis=0)
                            n_step_targets_opp = np.sum(np.multiply(np.asarray(opp_n_step_rewards[n:n+n_steps]),(n_step_gammas)),axis=0)

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
                        if D4PG:
                            default_prio = 5.0
                        else:
                            default_prio = 3.0
                        priorities = np.array([np.zeros(k_ensembles) for i in range(num_TA)])
                        priorities[:,current_ensembles] = 5.0
                        #print(current_ensembles)
                        if SIL:
                            SIL_priorities = np.ones(num_TA)*default_prio
                        exp_team = np.column_stack((np.transpose(team_n_step_obs[n]),
                                            team_n_step_acs[n],
                                            np.expand_dims(team_n_step_rewards[n], 1),
                                            np.transpose(n_step_next_ob_team),
                                            np.expand_dims([n_step_done_team for i in range(num_TA)], 1),
                                            np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_team, 1),
                                            np.expand_dims([team_n_step_ws[n] for i in range(num_TA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(num_TA)],1)))


                        exp_opp = np.column_stack((np.transpose(opp_n_step_obs[n]),
                                            opp_n_step_acs[n],
                                            np.expand_dims(opp_n_step_rewards[n], 1),
                                            np.transpose(n_step_next_ob_opp),
                                            np.expand_dims([n_step_done_opp for i in range(num_OA)], 1),
                                            np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_opp, 1),
                                            np.expand_dims([opp_n_step_ws[n] for i in range(num_OA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(num_TA)],1)))
                
                        exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)

                        if exps is None:
                            exps = torch.from_numpy(exp_comb)
                        else:
                            exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)
                    exp_i[int(ep_num[env_num].item())] += et_i
                    shared_exps[int(ep_num[env_num].item())][:len(exps)] = exps
                    ep_num[env_num] += 1
                    del exps
                    exps = None


                #############################################################################################################################################################
                # push exp to queue
                # log
                if ep_i > 1:
                    team_avg_rew = [np.asarray(team_n_step_rewards)[:,i].sum()/float(et_i) for i in range(num_TA)] # divide by time step?
                    team_cum_rew = [np.asarray(team_n_step_rewards)[:,i].sum() for i in range(num_TA)]
                    opp_avg_rew = [np.asarray(opp_n_step_rewards)[:,i].sum()/float(et_i) for i in range(num_TA)]
                    opp_cum_rew = [np.asarray(opp_n_step_rewards)[:,i].sum() for i in range(num_TA)]

                    #Added to Disable/Enable the team agents
                    if has_team_Agents:
                        team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(tkc/time_step)*100 for tkc in team_kickable_counter],
                                                            'possession_percentages': [(tpc/time_step)*100 for tpc in team_possession_counter],
                                                            'average_reward': team_avg_rew,
                                                            'cumulative_reward': team_cum_rew},
                                                            ignore_index=True)

                    #Added to Disable/Enable the opp agents
                    if has_opp_Agents:
                        opp_step_logger_df = opp_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.opp_team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(okc/time_step)*100 for okc in opp_kickable_counter],
                                                            'possession_percentages': [(opc/time_step)*100 for opc in opp_possession_counter],
                                                            'average_reward': opp_avg_rew,
                                                            'cumulative_reward': opp_cum_rew},
                                                            ignore_index=True)


                # Launch evaluation session
                if ep_i > 1 and ep_i % eval_after == 0 and evaluate:
                    thread.start_new_thread(launch_eval,(
                        [load_path + ("agent_%i/model_episode_%i.pth" % (i,ep_i)) for i in range(env.num_TA)], # models directory -> agent -> most current episode
                        eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation_ep" + str(ep_i),
                        7000,env.num_TA,env.num_OA,episode_length,device,use_viewer,))
                if halt.all(): # load when other process is loading buffer to make sure its not saving at the same time
                    #num_updates[env_num] += float(np.floor(exp_i/steps_per_update))
                    #exps.cpu()
                    ready[env_num] = 1
                    if np.random.uniform(0,1) > self_play_proba: # self_play_proba % chance loading self else load an old ensemble for opponent
                        maddpg.load_random_policy(side='opp',nagents = num_OA,models_path = load_path,load_same_agent=load_same_agent)
                    else:
                        maddpg.load_random_ensemble(side='opp',nagents = num_OA,models_path = ensemble_path,load_same_agent=load_same_agent)
                    current_ensembles = maddpg.load_random_ensemble(side='team',nagents=num_TA,models_path = ensemble_path,load_same_agent=load_same_agent) # use for per ensemble update counter
                    while halt.all():
                        time.sleep(0.1)
                    total_dim = (obs_dim_TA*2 + acs_dim + 5) + k_ensembles + 1
                    ep_num.copy_(torch.zeros_like(ep_num,requires_grad=False))
                    [s.copy_(torch.zeros(max_num_experiences,2*num_TA,total_dim)) for s in shared_exps[:int(ep_num[env_num].item())]] # done loading
                    #exp_i = 0 # reset experience builders
                    del exps
                    exps = None


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



if __name__ == "__main__":  
    mp.set_start_method('forkserver',force=True)
    seed = 912
    num_envs = 2
    port = 2000
    max_num_experiences = 500
    update_threads = []
    if True: # all options 
       # Parseargs --------------------------------------------------------------
        parser = argparse.ArgumentParser(description='Load port and log directory')
        parser.add_argument('-log_dir', type=str, default='log',
                        help='A name for log directory')
        parser.add_argument('-log', type=str, default='history',
                        help='A name for log file ')

        args = parser.parse_args()
        log_dir = args.log_dir
        history = args.log
            
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
        replay_memory_size = 300000
        episode_length = 500 # FPS
        untouched_time = 500
        burn_in_iterations = 500 # for time step
        burn_in_episodes = float(burn_in_iterations)/untouched_time
        deterministic = True

        # --------------------------------------
        # Team ---------------------------------
        num_TA = 2
        num_OA = 2
        num_TNPC = 0
        num_ONPC = 0
        acs_dim = 8
        offense_team_bin='helios10'
        defense_team_bin='helios11'  
        goalie = True
        team_rew_anneal_ep = 1500 # reward would be
        # hyperparams--------------------------
        batch_size = 128
        hidden_dim = int(512)

        tau = 0.001 # soft update rate

        steps_per_update = (2 * num_envs)
        number_of_updates = 0
        # exploration --------------------------
        explore = True
        final_OU_noise_scale = 0.1
        final_noise_scale = 0.1
        init_noise_scale = 1.00
        num_explore_episodes = 100 # Haus uses over 10,000 updates --
        multi_gpu = False

        # --------------------------------------
        #D4PG Options --------------------------
        D4PG = False
        gamma = 0.99 # discount
        Vmax = 40
        Vmin = -40
        N_ATOMS = 251
        DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
        n_steps = 5
        # n-step update size 
        # Mixed taqrget beta (0 = 1-step, 1 = MC update)
        initial_beta = 0.3
        final_beta =0.3
        num_beta_episodes = 1000000000
        if D4PG:
            a_lr = 0.0001 # actor learning rate
            c_lr = 0.001 # critic learning rate
        else:
            a_lr = 0.0001 # actor learning rate
            c_lr = 0.0001 # critic learning rate
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
        agents_x_min = -0.2
        agents_x_max = 0.2
        agents_y_min = -0.2
        agents_y_max = 0.2
        change_every_x = 1000000000
        change_agents_x = 0.01
        change_agents_y = 0.01
        change_balls_x = 0.01
        change_balls_y = 0.01
        # Self-play ----------------------------
        load_random_nets = True
        load_random_every = 100
        k_ensembles = 1
        current_ensembles = [0]*num_TA # initialize which ensembles we start with
        self_play_proba = 0.5
        load_same_agent = True # load same policy for all agents
        num_update_threads = num_TA
        if load_same_agent:
            num_update_threads = 1
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
        threads = []
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
    
    # dummy env that isn't used explicitly ergo used for dimensions
    env = HFO_env(num_TNPC = num_TNPC,num_TA=num_TA,num_OA=num_OA, num_ONPC=num_ONPC, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length, seed=seed, # create environment
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=65000,log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game, team_rew_anneal_ep=team_rew_anneal_ep,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=True,
                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, run_server=False, deterministic=deterministic)
    
    obs_dim_TA = env.team_num_features
    obs_dim_OA = env.opp_num_features

    # zip params for env processes
    HP = (action_level,feature_level,to_gpu,device,use_viewer,use_viewer_after,n_training_threads,rcss_log_game,hfo_log_game,num_episodes,replay_memory_size,
    episode_length,untouched_time,burn_in_iterations,burn_in_episodes, deterministic, num_TA,num_OA,num_TNPC,num_ONPC,offense_team_bin,defense_team_bin,goalie,team_rew_anneal_ep,
        batch_size,hidden_dim,a_lr,c_lr,tau,steps_per_update,explore,final_OU_noise_scale,final_noise_scale,init_noise_scale,num_explore_episodes,D4PG,gamma,Vmax,Vmin,N_ATOMS,
        DELTA_Z,n_steps,initial_beta,final_beta,num_beta_episodes,TD3,TD3_delay_steps,TD3_noise,I2A,decent_EM,EM_lr,obs_weight,rew_weight,ws_weight,rollout_steps,
        LSTM_hidden,imagination_policy_branch,SIL,SIL_update_ratio,critic_mod_act,critic_mod_obs,critic_mod_both,control_rand_init,ball_x_min,ball_x_max,
        ball_y_min,ball_y_max,agents_x_min,agents_x_max,agents_y_min,agents_y_max,change_every_x,change_agents_x,change_agents_y,change_balls_x,change_balls_y,
        load_random_nets,load_random_every,k_ensembles,current_ensembles,self_play_proba,save_nns,load_nets,initial_models,evaluate,eval_after,eval_episodes,
        LSTM,LSTM_PC,trace_length,hidden_dim_lstm,parallel_process,forward_pass,session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,t,time_step,discrete_action,
        has_team_Agents,has_opp_Agents,log_dir,obs_dim_TA,obs_dim_OA, acs_dim,max_num_experiences,load_same_agent,multi_gpu)

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
                                LSTM=LSTM, LSTM_PC=LSTM_PC, trace_length=trace_length, hidden_dim_lstm=hidden_dim_lstm,only_policy=False,multi_gpu=multi_gpu) 


    if first_save: # Generate list of ensemble networks
        file_path = ensemble_path
        maddpg.first_save(file_path,num_copies = k_ensembles)
        [maddpg.save_agent(load_path,0,i,load_same_agent = False) for i in range(num_TA)] 

        first_save = False

    team_replay_buffer = ReplayTensorBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM, LSTM_PC,k_ensembles,SIL)

    #Added to Disable/Enable the opp agents
        #initialize the replay buffer of size 10000 for number of opponent agent with their observations & actions 
    opp_replay_buffer = ReplayTensorBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM, LSTM_PC,k_ensembles,SIL)
    max_episodes_shared = 15
    update_session = 0
    processes = []
    total_dim = (obs_dim_TA*2 + acs_dim + 5) + k_ensembles + 1

    shared_exps = [[torch.zeros(max_num_experiences,2*num_TA,total_dim,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
    exp_indices = [[torch.tensor(0,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
    ep_num = torch.zeros(num_envs,requires_grad=False).share_memory_()

    halt = Variable(torch.tensor(0).byte()).share_memory_()
    ready = torch.zeros(num_envs,requires_grad=False).byte().share_memory_()
    update_counter = torch.zeros(num_envs,requires_grad=False).share_memory_()
    for i in range(num_envs):
        processes.append(mp.Process(target=run_envs, args=(seed + (i * 100), port + (i * 1000), shared_exps[i],exp_indices[i],HP,i,ready,halt,update_counter,(history+str(i)),ep_num)))
    
    for p in processes:
        p.start()

    iterations_per_push = 1
    #maddpg_pick = dill.dumps(maddpg)
    while True: # get experiences, update
        while((np.asarray([counter.item() for counter in ep_num]) < iterations_per_push).any()):
            time.sleep(0.1)
        halt.copy_(torch.tensor(1,requires_grad=False).byte())
        while not ready.all():
            time.sleep(0.1)
        for i in range(num_envs):
            #[team_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], :num_TA, :]) for j in range(int(ep_num[i].item()))]
            #[opp_replay_buffer.push(shared_exps[i][j][:exp_indices[i][j], num_TA:2*num_TA, :]) for j in range(int(ep_num[i].item()))]
            [team_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], :num_TA, :], shared_exps[i][j][:exp_indices[i][j], -num_TA:, :]))) for j in range(int(ep_num[i].item()))]
            [opp_replay_buffer.push(torch.cat((shared_exps[i][j][:exp_indices[i][j], -num_TA:, :], shared_exps[i][j][:exp_indices[i][j], :num_TA, :]))) for j in range(int(ep_num[i].item()))]
        # get num updates and reset counter
        # If the update rate is slower than the exp generation than this ratio will be greater than 1 when our experience tensor
        # is full (10,000 timesteps backlogged) so wait for updates to catch up
        print("Episode buffer/Max Shared memory at :",100*ep_num[0].item()/float(max_episodes_shared),"%")

        if (ep_num[0].item()/max_episodes_shared) >= 10:
            print("Training backlog (shared memory buffer full); halting experience generation until updates catch up")

            number_of_updates = int(update_counter.sum().item())
            threads = []
            if not load_same_agent:
                for a_i in range(maddpg.nagents_team):
                    threads.append(mp.Process(target=update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                        team_replay_buffer,opp_replay_buffer,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_PC,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu)))
            else:
                for a_i in range(1):
                    threads.append(mp.Process(target=update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                        team_replay_buffer,opp_replay_buffer,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_PC,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu)))



            [thr.start() for thr in threads]
            print("Launching update")
            [thr.join() for thr in threads]
            update_session +=1
            update_counter.copy_(torch.zeros(num_envs,requires_grad=False))


        for envs in exp_indices:
            for exp_i in envs:
                exp_i.copy_(torch.tensor(0,requires_grad=False))
        number_of_updates = int(update_counter.sum().item())
        update_counter.copy_(torch.zeros(num_envs,requires_grad=False))

        halt.copy_(torch.tensor(0,requires_grad=False).byte())
        ready.copy_(torch.zeros(num_envs,requires_grad=False).byte())

        training = (len(team_replay_buffer) >= batch_size)

        if training:
            
            threads = []
            if not load_same_agent:
                for a_i in range(maddpg.nagents_team):
                    threads.append(mp.Process(target=update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                        team_replay_buffer,opp_replay_buffer,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_PC,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu)))
            else:
                for a_i in range(1):
                    threads.append(mp.Process(target=update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                        team_replay_buffer,opp_replay_buffer,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_PC,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu)))

            [thr.start() for thr in threads]
            print("Launching update")
            [thr.join() for thr in threads]
            update_session +=1

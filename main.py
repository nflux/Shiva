import itertools
import random
import datetime
import os, sys
import csv
import argparse
import numpy as np
import utils.misc as misc
from torch import Tensor
from HFO import hfo
import time
import threading
import _thread as thread
import torch
from pathlib import Path
from torch.autograd import Variable
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MADDPG
from rc_env import rc_env, run_envs
from trainer import launch_eval
import torch.multiprocessing as mp
import dill
import collections
from multiprocessing import Pool
import gc
import algorithms.updates as up
import configparser

def str_to_bool(s):
    return s == 'True'

if __name__ == "__main__":
    sys_args = sys.argv[:]

    if len(sys_args) == 4:
        env_args = ' '.join(sys_args[1:3])
        conf_path = os.getcwd() + '/configs/' + sys_args[3]
    else:
        print('Invalid number of args provided')
        exit(0)

    config = configparser.ConfigParser()

    if env_args == "-env rc":
        mp.set_start_method('forkserver',force=True)
        config.read(conf_path)

        # ENV_PARAMS
        EP = config['ENV_PARAMS']
        seed = int(EP['SEED'])
        num_envs = int(EP['NUM_ENVS'])
        port = int(EP['PORT'])
        max_num_experiences = int(EP['MNE'])
        log_dir = EP['LOG']
        history = EP['HISTORY']
        action_level = EP['AL']
        feature_level = EP['FL']
        use_viewer = str_to_bool(EP['UV'])
        rcss_log_game = str_to_bool(EP['RCSS_LOG'])
        hfo_log_game = str_to_bool(EP['HFO_LOG'])
        num_episodes = int(EP['NUM_EP'])
        episode_length = int(EP['EP_LENGTH'])
        untouched_time = int(EP['UNTOUCHED'])
        deterministic = str_to_bool(EP['DETERM'])
        burn_in_iterations = int(EP['BURN_IN'])
        burn_in_episodes = float(burn_in_iterations)/untouched_time
        record = str_to_bool(EP['RECORD_LIB'])
        record_server = str_to_bool(EP['RECORD_SERV'])
        num_TA = int(EP['NUM_LEFT'])
        num_OA = int(EP['NUM_RIGHT'])
        num_TNPC = int(EP['NUM_L_BOT'])
        num_ONPC = int(EP['NUM_R_BOT'])
        acs_dim = int(EP['AC_DIM'])
        offense_team_bin = EP['LEFT_BIN']
        defense_team_bin = EP['RIGHT_BIN']
        goalie = str_to_bool(EP['GOALIE'])
        team_rew_anneal_ep = int(EP['REWARD_ANNEAL'])

        # INIT_ENVS
        IE = config['INIT_ENVS']
        # Control Random Initilization of Agents and Ball
        control_rand_init = str_to_bool(IE['SELF'])
        ball_x_min = float(IE['BALL_X_MIN'])
        ball_x_max = float(IE['BALL_X_MAX'])
        ball_y_min = float(IE['BALL_Y_MIN'])
        ball_y_max = float(IE['BALL_Y_MAX'])
        # agents posititions are currently hard coded on the server
        agents_x_min = -0.2 
        agents_x_max = 0.2
        agents_y_min = -0.2
        agents_y_max = 0.2
        # Below change_xxx are not used currently
        change_every_x = 1000000000
        change_agents_x = 0.01
        change_agents_y = 0.01
        change_balls_x = 0.01
        change_balls_y = 0.01

        # HPS_BASE
        HP = config['HPS_BASE']
        batch_size = int(HP['BATCH_SIZE'])
        hidden_dim = int(HP['HIDDEN_DIM'])
        tau = float(HP['TAU'])
        number_of_updates = int(HP['NUM_UPDATES'])
        explore = str_to_bool(HP['EXPLORE'])
        final_OU_noise_scale = float(HP['OU_NOISE'])
        final_noise_scale = float(HP['NOISE'])
        init_noise_scale = float(HP['INIT_NOISE'])
        num_explore_episodes = int(HP['NUM_EXP_EPS'])

        # D4PG
        HP = config['D4PG']
        D4PG = str_to_bool(HP['SELF'])
        gamma = float(HP['GAMMA']) # discount
        Vmax = int(HP['VMAX'])
        Vmin = int(HP['VMIN'])
        N_ATOMS = int(HP['N_ATOMS'])
        DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
        if D4PG:
            a_lr = 0.0001 # actor learning rate
            c_lr = 0.001 # critic learning rate
        else:
            freeze_actor = 0.0
            freeze_critic = 0.0

            a_lr = 0.0002 # actor learning rate
            c_lr = 0.0001 # critic learning rate
        
        # N-Step
        HP = config['N_STEP']
        n_steps = int(HP['SELF'])
        # Mixed taqrget beta (0 = 1-step, 1 = MC update)
        initial_beta = float(HP['INIT_BETA'])
        final_beta = float(HP['FINAL_BETA'])
        num_beta_episodes = int(HP['NUM_BETA_EPS'])

        # TD3
        HP = config['TD3']
        TD3 = str_to_bool(HP['SELF'])
        TD3_delay_steps = int(HP['TD3_DELAY'])
        TD3_noise = float(HP['TD3_NOISE'])

        # Buffer Params
        BP = config['BUFFER_PARAMS']
        replay_memory_size = int(BP['REPLAY_SIZE'])
        pt_memory = int(BP['PT_MEM'])

        # Pretrain
        PT = config['PRETRAIN']
        pretrain = str_to_bool(PT['SELF'])
        use_pretrain_data = str_to_bool(PT['USE_PT_DATA'])
        test_imitation = str_to_bool(PT['TEST_IMI'])  # After pretrain, infinitely runs the current pretrained policy
        pt_update_cycles = int(PT['PT_CYCLES'])
        pt_inject_proba = float(PT['PT_INJECT_PROB'])
        init_pt_inject_proba = float(PT['INIT_PT_PROB'])
        final_pt_inject_proba = float(PT['FINAL_PT_PROB'])
        pt_inject_anneal_ep = int(PT['PT_ANNEAL_EP'])
        play_agent2d = str_to_bool(PT['AGENT2D'])
        bl_agent2d = str_to_bool(PT['LEFT_AGENT2D'])
        use_preloaded_agent2d = str_to_bool(PT['PRELOAD_AGENT2D'])
        preload_agent2d_path = ""
        num_buffers = int(PT['NUM_BUFFS'])
        pt_total_memory = pt_memory*num_buffers

        # I2A
        HP = config['I2A']
        I2A = str_to_bool(HP['SELF'])
        EM_lr = float(HP['EM_LR'])
        obs_weight = float(HP['OBS_W'])
        rew_weight = float(HP['REW_W'])
        ws_weight = float(HP['WS_W'])
        rollout_steps = int(HP['ROLL_STEPS'])
        LSTM_hidden= int(HP['LSTM_HIDD'])
        imagination_policy_branch = str_to_bool(HP['IMAG_POL_BRANCH'])

        # SIL
        HP = config['SIL']
        SIL = str_to_bool(HP['SELF'])
        SIL_update_ratio = int(HP['UPDATE_RATIO'])

        # CENT Q
        HP =  config['CENT_Q']
        critic_mod = str_to_bool(HP['SELF'])
        critic_mod_act = str_to_bool(HP['CRIT_AC'])
        critic_mod_obs = str_to_bool(HP['CRIT_OBS'])
        critic_mod_both = ((critic_mod_act == False) and (critic_mod_obs == False) and critic_mod)

        # Self Play
        SP = config['SELF_PLAY']
        load_random_nets = str_to_bool(SP['LOAD_RAND_NETS'])
        load_random_every = int(SP['LOAD_RAND_EVERY'])
        k_ensembles = int(SP['K_ENS'])
        current_ensembles = [0]*num_TA # initialize which ensembles we start with
        self_play_proba = float(SP['SELF_PLAY_PROB'])
        load_same_agent = str_to_bool(SP['LOAD_SAME_AGENT']) # load same policy for all agents
        push_only_left = str_to_bool(SP['PUSH_ONLY_LEFT'])
        num_update_threads = num_TA
        if load_same_agent:
            num_update_threads = 1
        
        # Save/Load
        SL = config['SAVE_LOAD']
        save_nns = str_to_bool(SL['SAVE_NNS'])
        ep_save_every = int(SL['EP_SAVE_EVERY']) # episodes
        load_nets = str_to_bool(SL['LOAD_NETS']) # load previous sessions' networks from file for initialization
        initial_models = ["training_sessions/1_11_8_1_vs_1/ensemble_models/ensemble_agent_0/model_0.pth"]
        first_save = str_to_bool(SL['FIRST_SAVE']) # build model clones for ensemble
        preload_model = str_to_bool(SL['PRELOAD_MODEL'])
        preload_path = "agent2d/model_0.pth"

        # Evaluation
        EVAL = config['EVAL']
        evaluate = str_to_bool(EVAL['SELF'])
        eval_after = int(EVAL['EVAL_AFTER'])
        eval_episodes = int(EVAL['EVAL_EPS'])

        # LSTM
        HP = config['LSTM']
        LSTM = str_to_bool(HP['LSTM_CRIT']) # Critic only
        LSTM_policy = str_to_bool(HP['LSTM_POL']) # Policy
        hidden_dim_lstm = int(HP['HIDDEN_DIM'])
        lstm_burn_in = int(HP['BURN_IN'])
        if LSTM:
            seq_length = int(HP['SEQ_LENGTH']) # Must be divisible by 2 b/c of overlap formula
            overlap = int(seq_length/2)
        else:
            seq_length = 0
            overlap = 0
        if seq_length % 2 != 0:
            print('Seq length must be divisible by 2')
            exit(0)
        
        # Misc
        MISC = config['MISC']
        USE_CUDA = str_to_bool(MISC['CUDA'])
        if USE_CUDA:
            device = 'cuda'
            to_gpu = True
        else:
            to_gpu = False
            device = 'cpu'
        n_training_threads = int(MISC['NUM_THREADS'])
        multi_gpu = str_to_bool(MISC['MULTI_GPU'])
        data_parallel = str_to_bool(MISC['DATA_PARALLEL'])
        # optimizations
        parallel_process = str_to_bool(MISC['PARALLEL_PROC'])
        forward_pass = str_to_bool(MISC['FORWARD_PASS'])
        preprocess = str_to_bool(MISC['PREPROCESS'])
        zero_critic = str_to_bool(MISC['ZERO_CRIT'])
        cent_critic = str_to_bool(MISC['CENT_CRIT'])

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
        misc.prep_session(session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,log_dir,num_TA) # Generates directories and files for the session

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
        
    # dummy env that isn't used explicitly ergo used for dimensions
    env = rc_env(num_TNPC = num_TNPC,num_TA=num_TA,num_OA=num_OA, num_ONPC=num_ONPC, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length, seed=seed, # create environment
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=65000,log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game, team_rew_anneal_ep=team_rew_anneal_ep,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=False,
                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, run_server=False, deterministic=deterministic)
    
    obs_dim_TA = env.team_num_features
    obs_dim_OA = env.opp_num_features
    # zip params for env processes
    HP = (action_level,feature_level,to_gpu,device,use_viewer,n_training_threads,rcss_log_game,hfo_log_game,num_episodes,replay_memory_size,
    episode_length,untouched_time,burn_in_iterations,burn_in_episodes, deterministic, num_TA,num_OA,num_TNPC,num_ONPC,offense_team_bin,defense_team_bin,goalie,team_rew_anneal_ep,
        batch_size,hidden_dim,a_lr,c_lr,tau,explore,final_OU_noise_scale,final_noise_scale,init_noise_scale,num_explore_episodes,D4PG,gamma,Vmax,Vmin,N_ATOMS,
        DELTA_Z,n_steps,initial_beta,final_beta,num_beta_episodes,TD3,TD3_delay_steps,TD3_noise,I2A,EM_lr,obs_weight,rew_weight,ws_weight,rollout_steps,
        LSTM_hidden,imagination_policy_branch,SIL,SIL_update_ratio,critic_mod_act,critic_mod_obs,critic_mod_both,control_rand_init,ball_x_min,ball_x_max,
        ball_y_min,ball_y_max,agents_x_min,agents_x_max,agents_y_min,agents_y_max,change_every_x,change_agents_x,change_agents_y,change_balls_x,change_balls_y,
        load_random_nets,load_random_every,k_ensembles,current_ensembles,self_play_proba,save_nns,load_nets,initial_models,evaluate,eval_after,eval_episodes,
        LSTM, LSTM_policy,seq_length,hidden_dim_lstm,lstm_burn_in,overlap,parallel_process,forward_pass,session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,t,time_step,discrete_action,
        log_dir,obs_dim_TA,obs_dim_OA, acs_dim,max_num_experiences,load_same_agent,multi_gpu,data_parallel,play_agent2d,use_preloaded_agent2d,preload_agent2d_path,
        bl_agent2d,preprocess,zero_critic,cent_critic, record, record_server)



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
                                rollout_steps = rollout_steps,LSTM_hidden=LSTM_hidden,
                                imagination_policy_branch = imagination_policy_branch,critic_mod_both=critic_mod_both,
                                critic_mod_act=critic_mod_act, critic_mod_obs= critic_mod_obs,
                                LSTM=LSTM, LSTM_policy=LSTM_policy,seq_length=seq_length, hidden_dim_lstm=hidden_dim_lstm, lstm_burn_in=lstm_burn_in,overlap=overlap,
                                only_policy=False,multi_gpu=multi_gpu,data_parallel=data_parallel,preprocess=preprocess,zero_critic=zero_critic,cent_critic=cent_critic) 

    if multi_gpu:
        maddpg.torch_device = torch.device("cuda:0")
    
    if to_gpu:
        maddpg.device = 'cuda'
    maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
    
    if first_save: # Generate list of ensemble networks
        file_path = ensemble_path
        maddpg.first_save(file_path,num_copies = k_ensembles)
        [maddpg.save_agent(load_path,0,i,load_same_agent = False,torch_device=maddpg.torch_device) for i in range(num_TA)] 
        if preload_model:
            maddpg.load_team(side='team',models_path=preload_path,nagents=num_TA)
            maddpg.first_save(file_path,num_copies = k_ensembles) 


        first_save = False
    
    prox_item_size = num_TA*(2*obs_dim_TA + 2*acs_dim)
    team_replay_buffer = ReplayBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM, seq_length,overlap,hidden_dim_lstm,k_ensembles, prox_item_size, SIL)

    #Added to Disable/Enable the opp agents
        #initialize the replay buffer of size 10000 for number of opponent agent with their observations & actions 
    opp_replay_buffer = ReplayBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM, seq_length,overlap,hidden_dim_lstm,k_ensembles, prox_item_size, SIL)
    max_episodes_shared = 30
    processes = []
    total_dim = (obs_dim_TA + acs_dim + 5) + k_ensembles + 1 + (hidden_dim_lstm*4) + prox_item_size

    shared_exps = [[torch.zeros(max_num_experiences,2*num_TA,total_dim,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
    exp_indices = [[torch.tensor(0,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
    ep_num = torch.zeros(num_envs,requires_grad=False).share_memory_()

    halt = Variable(torch.tensor(0).byte()).share_memory_()
    ready = torch.zeros(num_envs,requires_grad=False).byte().share_memory_()
    update_counter = torch.zeros(num_envs,requires_grad=False).share_memory_()
    for i in range(num_envs):
        processes.append(mp.Process(target=run_envs, args=(seed + (i * 100), port + (i * 1000), shared_exps[i],exp_indices[i],HP,i,ready,halt,update_counter,(history+str(i)),ep_num)))
    


    
    if pretrain or use_pretrain_data:
        # ------------------------------------ Start Pretrain --------------------------------------------
        pt_trbs = [ReplayBuffer(pt_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size, LSTM,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain) for _ in range(num_buffers)]
        pt_orbs = [ReplayBuffer(pt_memory , num_TA,
                                            obs_dim_TA,acs_dim,batch_size, LSTM,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain) for _ in range(num_buffers)]
        # ------------ Load in shit from csv into buffer ----------------------
        pt_threads = []
        print("Load PT Buffers")
        all_l = []
        all_r = []
        all_s = []
        for i in range(1,num_buffers+1):
            left_files = []
            right_files = []
            
            if os.path.isdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_%i' % ((i+1)*1000)):
                team_files = os.listdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_%i' % ((i+1)*1000))
                left_files = [os.getcwd() + '/pretrain/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + f for f in team_files if '_left_' in f]
                right_files = [os.getcwd() + '/pretrain/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + f for f in team_files if '_right_' in f]
                status_file = os.getcwd() + '/pretrain/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + 'log_status.csv'
            else:
                print('log directory DNE')
                exit(0)
            all_l.append(left_files)
            all_r.append(right_files)
            all_s.append(status_file)
    
    
        second_args = [(num_TA,obs_dim_TA,acs_dim,pt_trbs[i],pt_orbs[i],episode_length,n_steps,gamma,D4PG,SIL,k_ensembles,push_only_left,LSTM_policy,prox_item_size) for i in range(num_buffers)]
        with Pool() as pool:
            pt_rbs = pool.starmap(misc.load_buffer, zip(all_l,all_r,all_s, second_args))
        pt_trbs,pt_orbs = list(zip(*pt_rbs))
        del pt_rbs
        for i in range(num_buffers):
            print("Length of RB_",i,": ",len(pt_trbs[i]))
            
        #team_replay_buffer = pt_trbs[0]
        #opp_replay_buffer = pt_orbs[0]
        
        pt_trb = ReplayBuffer(pt_total_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size,LSTM,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain)
        pt_orb = ReplayBuffer(pt_total_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size,LSTM,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain)
        for j in range(num_buffers):
            if not LSTM_policy:
                pt_trb.merge_buffer(pt_trbs[j])
                pt_orb.merge_buffer(pt_orbs[j])
            else:
                pt_trb.merge_buffer_LSTM(pt_trbs[j])
                pt_orb.merge_buffer_LSTM(pt_orbs[j])
            print("Merging Buffer: ",j)
        del pt_trbs
        del pt_orbs
        gc.collect()
        time.sleep(5)
        print("Length of centralized buffer: ", len(pt_trb))
    if pretrain:

        # ------------ Done loading shit --------------------------------------
        # ------------ Pretrain actor/critic same time---------------
        #define/update the noise used for exploration

        maddpg.scale_noise(0.0)
        maddpg.reset_noise()
        maddpg.scale_beta(1)
        update_session = 999999

        for u in range(pt_update_cycles):
            print("PT Update Cycle: ",u)
            print("PT Completion: ",(u*100.0)/(float(pt_update_cycles)),"%")
            rb_i = np.random.randint(num_buffers)


            threads = []
            for a_i in range(1):
                threads.append(mp.Process(target=up.imitation_thread,args=(a_i,to_gpu,len(pt_trb),batch_size,
                    pt_trb,pt_orb,number_of_updates,
                    load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_policy,seq_length,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu,session_path,data_parallel,lstm_burn_in)))
            [thr.start() for thr in threads]
            
            agentID = 0
            buffer_size = len(pt_trb)

            number_of_updates = 5000
            batches_to_sample = 50
            if len(pt_trb) < batch_size*(batches_to_sample):
                batches_to_sample = 1
            priorities = [] 
            start = time.time()
            for ensemble in range(k_ensembles):

                maddpg.load_same_ensembles(ensemble_path,ensemble,maddpg.nagents_team,load_same_agent=load_same_agent)

                #start = time.time()
                for up in range(number_of_updates):
                    offset = up % batches_to_sample

                    if not load_same_agent:
                        inds = pt_trb.get_PER_inds(agentID,batch_size,ensemble)
                    else:
                        if up % batches_to_sample == 0:
                            m = 0
                            #inds = pt_trb[rb_i].get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                            inds = np.random.choice(np.arange(len(pt_trb)), size=batch_size*batches_to_sample, replace=False)

                            #if len(priorities) > 1:
                                #for c,p in enumerate(priorities):
                                #    pt_trb.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                            priorities = [] 


                    # FOR THE LOVE OF GOD DONT USE TORCH TO GET INDICES

                    if LSTM:
                        team_sample = pt_trb.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],to_gpu=to_gpu,device=maddpg.torch_device)
                        opp_sample = pt_orb.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],to_gpu=to_gpu,device=maddpg.torch_device)
                        priorities=maddpg.pretrain_critic_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',load_same_agent=load_same_agent,lstm_burn_in=lstm_burn_in)
                        #print("Use pretrain function")
                        #pt_trb.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)

                        if not load_same_agent:
                            print("No implementation")
                    else:
                        team_sample = pt_trb.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                    to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                        opp_sample = pt_orb.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                    to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                        if not load_same_agent:
                            print("No implementation")
                        else:
                            
                            priorities.append(maddpg.pretrain_critic_MC(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,session_path=session_path))
                            #priorities.append(maddpg.pretrain_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,session_path=session_path))
                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                maddpg.update_agent_targets(0,number_of_updates/10)
                    if SIL:
                        for i in range(SIL_update_ratio):
                            inds = pt_trb.get_SIL_inds(agentID=m,batch_size=batch_size)
                            team_sample = pt_trb.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                            opp_sample = pt_orb.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                            priorities = maddpg.SIL_update(team_sample, opp_sample, agentID, 'team') # 
                            pt_trb.update_SIL_priorities(agentID=m,inds = inds, prio=priorities)
                #for c,p in enumerate(priorities):
                #    if c != (len(priorities)-1):
                #        pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):(batch_size*(c+1))+(-batch_size*batches_to_sample)], prio=p,k = ensemble) 
                #    else:
                #        pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):], prio=p,k = ensemble) 
                #print(time.time()-start)
                if not load_same_agent:
                    maddpg.update_agent_targets(agentID,number_of_updates)
                    maddpg.save_agent(load_path,update_session,agentID)
                    maddpg.save_ensemble(ensemble_path,ensemble,agentID)
                else:
                    maddpg.update_agent_hard_critic(0)
                    print(time.time()-start,"<-- Critic Update Cycle")

                    [thr.join() for thr in threads]
                    maddpg.load_ensemble_policy(ensemble_path,ensemble,0) # load only policy from updated policy thread
                    maddpg.update_agent_hard_policy(agentID=0)
                    [maddpg.save_agent(load_path,update_session,i,load_same_agent,maddpg.torch_device) for i in range(num_TA)]
                    [maddpg.save_ensemble(ensemble_path,ensemble,i,load_same_agent,maddpg.torch_device) for i in range(num_TA)]
                print(time.time()-start,"<-- Full Cycle")






            maddpg.update_hard_policy() 
            maddpg.update_hard_critic()
            maddpg.save_agent2d(load_path,0,load_same_agent,maddpg.torch_device)


    # -------------Done pretraining actor/critic ---------------------------------------------
    maddpg.save_agent2d(load_path,0,load_same_agent,maddpg.torch_device)
    [maddpg.save_ensemble(ensemble_path,0,i,load_same_agent,maddpg.torch_device) for i in range(num_TA)] # Save agent2d into ensembles

    maddpg.scale_beta(initial_beta) 

    for p in processes: # Starts environments
        p.start()
    iterations_per_push = 1
    update_session = 0

    #maddpg_pick = dill.dumps(maddpg)
    while True: # get experiences, update
        cycle = 0
        while((np.asarray([counter.item() for counter in ep_num]) < iterations_per_push).any()):
            time.sleep(0.1)
        halt.copy_(torch.tensor(1,requires_grad=False).byte())
        while not ready.all():
            time.sleep(0.1)
        misc.push(team_replay_buffer,opp_replay_buffer,num_envs,shared_exps,exp_indices,num_TA,ep_num,seq_length,LSTM,push_only_left)
            
        # get num updates and reset counter
        # If the update rate is slower than the exp generation than this ratio will be greater than 1 when our experience tensor
        # is full (10,000 timesteps backlogged) so wait for updates to catch up
        print("Episode buffer/Max Shared memory at :",100*ep_num[0].item()/float(max_episodes_shared),"%")

        if (ep_num[0].item()/max_episodes_shared) >= 10:
            print("Training backlog (shared memory buffer full); halting experience generation until updates catch up")

            number_of_updates = 500
            threads = []
            if not load_same_agent:
                print("Implementation out of date (load same agent)")
                for a_i in range(maddpg.nagents_team):
                    threads.append(mp.Process(target=up.update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                        team_replay_buffer,opp_replay_buffer,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_policy,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu,session_path,data_parallel,lstm_burn_in)))
            else:                    
                for a_i in range(1):
                    threads.append(mp.Process(target=up.update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                                                                         team_replay_buffer,opp_replay_buffer,number_of_updates,
                                                                         load_path,update_session,ensemble_path,forward_pass,
                                                                         LSTM,LSTM_policy,k_ensembles,
                                                                         SIL,SIL_update_ratio,num_TA,load_same_agent,
                                                                         multi_gpu,session_path,data_parallel,lstm_burn_in)))
            print("Launching update")
            start = time.time()
            [thr.start() for thr in threads]
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

                            
        maddpg.load_same_ensembles(ensemble_path,0,maddpg.nagents_team,load_same_agent=load_same_agent)        


        training = (len(team_replay_buffer) >= batch_size)
        if training:
            threads = []
            pt_inject_proba = max(0.0, 1.0*pt_inject_anneal_ep - update_session) / (pt_inject_anneal_ep)
            pt_inject_proba = final_pt_inject_proba + (init_pt_inject_proba - final_pt_inject_proba) * pt_inject_proba
            PT = (np.random.uniform(0,1) < pt_inject_proba) and (pretrain or use_pretrain_data)
            print("Update Session:",update_session)
            if PT or use_pretrain_data:

                print("PT sample probability:",pt_inject_proba)
                print("Using PT sample:",PT)

        
            if PT:
                rb_i = np.random.randint(num_buffers)

                for a_i in range(1):
                    threads.append(mp.Process(target=up.update_thread,args=(a_i,to_gpu,len(pt_trb),batch_size,
                        pt_trb,pt_orb,number_of_updates,
                        load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_policy,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu,session_path,data_parallel,lstm_burn_in)))
            else:
                if update_session > 75:
                    for a_i in range(1):
                        threads.append(mp.Process(target=up.update_thread,args=(a_i,to_gpu,len(team_replay_buffer),batch_size,
                                                                             team_replay_buffer,opp_replay_buffer,number_of_updates,
                                                                             load_path,update_session,
                                                                             ensemble_path,forward_pass,LSTM,LSTM_policy,
                                                                             k_ensembles,SIL,SIL_update_ratio,num_TA,
                                                                             load_same_agent,multi_gpu,session_path,
                                                                             data_parallel,lstm_burn_in)))
                [thr.start() for thr in threads]
            print("Launching update")
            start = time.time()

            update_session +=1
            agentID = 0
            buffer_size = len(team_replay_buffer)

            number_of_updates = 250
            batches_to_sample = 50
            if len(team_replay_buffer) < batch_size*(batches_to_sample):
                batches_to_sample = 1

            priorities = [] 

            for ensemble in range(k_ensembles):
                for up in range(number_of_updates):
                    offset = up % batches_to_sample
                    m = 0


                    if not load_same_agent:
                        inds = team_replay_buffer.get_PER_inds(agentID,batch_size,ensemble)
                    else:
                        if up % batches_to_sample == 0:
                            if PT:
                                #inds = pt_trb.get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                                inds = np.random.choice(np.arange(len(pt_trb)), size=batch_size*batches_to_sample, replace=False)

                            else:
                                #inds = team_replay_buffer.get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                                inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size*batches_to_sample, replace=False)
                            #if len(priorities) > 1:
                            #    for c,p in enumerate(priorities):
                            #        if PT:
                            #            pt_trb.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                            #        else:
                            #            team_replay_buffer.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                            priorities = [] 

                    #inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)

                    # FOR THE LOVE OF GOD DONT USE TORCH TO GET INDICES

                    if LSTM:
                        team_sample = team_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                                    to_gpu=to_gpu, device=maddpg.torch_device)
                        opp_sample = opp_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                                    to_gpu=to_gpu, device=maddpg.torch_device)

                        if not load_same_agent:
                            print("No implementation")

                        else:
                            train_actor = (len(team_replay_buffer) > 10000) and False # and (update_session % 2 == 0)
                            train_critic = (len(team_replay_buffer) > batch_size)
                            if train_critic:
                                priorities.append(maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,critic=True,policy=False,session_path=session_path,lstm_burn_in=lstm_burn_in))
                            if train_actor: # only update actor once 1 mill
                                _ = maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,
                                                                            load_same_agent=load_same_agent,critic=False,policy=True,session_path=session_path)

                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                if train_critic and not train_actor:
                                    maddpg.update_agent_critic(0,number_of_updates/10)
                                elif train_critic and train_actor:
                                    maddpg.update_agent_targets(0,number_of_updates/10)
                    else:
                        if PT:
                            
                            team_sample = pt_trb.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                        to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                            opp_sample = pt_orb.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                        to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                        else:
                            team_sample = team_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                        to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                            opp_sample = opp_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                        to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                        if not load_same_agent:
                            print("no implementation")
                        else:
                            train_actor = (len(team_replay_buffer) > 1000) and False # and (update_session % 2 == 0)
                            train_critic = (len(team_replay_buffer) > 1000)
                            if train_critic:
                                priorities.append(maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,critic=True,policy=False,session_path=session_path))
                            if train_actor: # only update actor once 1 mill
                                _ = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,critic=False,policy=True,session_path=session_path)

                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                if train_critic and not train_actor:
                                    maddpg.update_agent_critic(0,number_of_updates/10)
                                elif train_critic and train_actor:
                                    maddpg.update_agent_targets(0,number_of_updates/10)
                    if SIL:
                        for i in range(SIL_update_ratio):
                            inds = team_replay_buffer.get_SIL_inds(agentID=m,batch_size=batch_size)
                            team_sample = team_replay_buffer.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                            opp_sample = opp_replay_buffer.sample(inds,
                                                    to_gpu=to_gpu,norm_rews=False)
                            priorities = maddpg.SIL_update(team_sample, opp_sample, agentID, 'team') # 
                            team_replay_buffer.update_SIL_priorities(agentID=m,inds = inds, prio=priorities)
                #if PT:
                #    for c,p in enumerate(priorities):
                #        if c != (len(priorities)-1):
                #            pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):(batch_size*(c+1))+(-batch_size*batches_to_sample)], prio=p,k = ensemble) 
                #        else:
                #            pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):], prio=p,k = ensemble) 
                #else:
                #    for c,p in enumerate(priorities):
                #        if c != (len(priorities)-1):
                #            team_replay_buffer.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):(batch_size*(c+1))+(-batch_size*batches_to_sample)], prio=p,k = ensemble) 
                #        else:
                #            team_replay_buffer.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):], prio=p,k = ensemble) 
                
                #print(time.time()-start)
                if not load_same_agent:
                    maddpg.update_agent_targets(agentID,number_of_updates)
                    maddpg.save_agent(load_path,update_session,agentID,torch_device=maddpg.torch_device)
                    maddpg.save_ensemble(ensemble_path,ensemble,agentID,torch_device=maddpg.torch_device)
                else:
                    maddpg.update_agent_targets(0,number_of_updates/10)
                    print(time.time()-start,"<-- Critic Update Cycle")

                    [thr.join() for thr in threads]
                    maddpg.load_ensemble_policy(ensemble_path,ensemble,0) # load only policy from updated policy thread
                    [maddpg.save_agent(load_path,update_session,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
                    [maddpg.save_ensemble(ensemble_path,ensemble,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
                print(time.time()-start,"<-- Full Cycle")
                cycle += 1


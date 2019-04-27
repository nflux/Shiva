import random
import os, sys
# import csv
import argparse
import numpy as np
import utils.misc as misc
import time
import torch
from pathlib import Path
import utils.buffers as buff
import algorithms.maddpg as mad_algo
import envs.multi_envs as menvs
# from trainer import launch_eval
import torch.multiprocessing as mp
from multiprocessing import Pool
import gc
import algorithms.updates as updates
import configparser
import argparse
import config

def parseArgs():
    parser =  argparse.ArgumentParser('Team Shiva')
    parser.add_argument('--env', type=str, default='rc',
                        help='type of environment')
    parser.add_argument('--conf', type=str, default='rc_test.ini')

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    config_parse = configparser.SafeConfigParser()

    conf_path = os.getcwd() + '/configs/' + args.conf

    # if args.env == 'rc':
    mp.set_start_method('forkserver',force=True)
    config_parse.read(conf_path)
    config = config.RoboConfig(config_parse)

    env = menvs.RoboEnvs(config)
    maddpg = mad_algo.init(config, env.template_env)
    

    if config.pt or config.use_pt_data:
        # ------------------------------------ Start Pretrain --------------------------------------------
        pt_trbs = [ReplayBufferLSTM(pt_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain) for _ in range(num_buffers)]
        pt_orbs = [ReplayBufferLSTM(pt_memory , num_TA,
                                            obs_dim_TA,acs_dim,batch_size,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain) for _ in range(num_buffers)]
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
        
        pt_trb = ReplayBufferLSTM(pt_total_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain)
        pt_orb = ReplayBufferLSTM(pt_total_memory , num_TA,
                                                obs_dim_TA,acs_dim,batch_size,seq_length,overlap,hidden_dim_lstm,k_ensembles,prox_item_size,SIL,pretrain=pretrain)
        for j in range(num_buffers):
            if not LSTM_policy:
                pt_trb.merge_buffer(pt_trbs[j])
                pt_orb.merge_buffer(pt_orbs[j])
            else:
                pt_trb.merge_buffer(pt_trbs[j])
                pt_orb.merge_buffer(pt_orbs[j])
            print("Merging Buffer: ",j)
        del pt_trbs
        del pt_orbs
        gc.collect()
        time.sleep(5)
        print("Length of centralized buffer: ", len(pt_trb))

    if config.pt:

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
                threads.append(mp.Process(target=updates.imitation_thread,args=(a_i,to_gpu,len(pt_trb),batch_size,
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
                        team_sample = pt_trb.sample(inds[batch_size*offset:batch_size*(offset+1)],to_gpu=to_gpu,device=maddpg.torch_device)
                        opp_sample = pt_orb.sample(inds[batch_size*offset:batch_size*(offset+1)],to_gpu=to_gpu,device=maddpg.torch_device)
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
    maddpg.save_agent2d(config.load_path,0,config.load_same_agent,maddpg.torch_device)
    [maddpg.save_ensemble(config.ensemble_path,0,i,config.load_same_agent,maddpg.torch_device) for i in range(config.num_left)] # Save agent2d into ensembles

    maddpg.scale_beta(config.init_beta) 

    # for p in processes: # Starts environments
    #     p.start()
    env.run()
    iterations_per_push = 1
    update_session = 0
    print('This gets here')
    time.sleep(30)

    while True: # get experiences, update
        cycle = 0
        while((np.asarray([counter.item() for counter in env.ep_num]) < iterations_per_push).any()):
            time.sleep(0.1)
        env.halt.copy_(torch.tensor(1,requires_grad=False).byte())
        while not env.ready.all():
            time.sleep(0.1)
        misc.push(env.team_replay_buffer,env.opp_replay_buffer,config.num_envs,env.shared_exps,env.exp_indices,config.num_left,env.ep_num,config.seq_length,config.lstm_crit,config.push_only_left)
            
        # get num updates and reset counter
        # If the update rate is slower than the exp generation than this ratio will be greater than 1 when our experience tensor
        # is full (10,000 timesteps backlogged) so wait for updates to catch up
        print("Episode buffer/Max Shared memory at :",100*env.ep_num[0].item()/float(env.max_episodes_shared),"%")

        if (env.ep_num[0].item()/env.max_episodes_shared) >= 10:
            print("Training backlog (shared memory buffer full); halting experience generation until updates catch up")

            number_of_updates = 500
            threads = []
            if not config.load_same_agent:
                print("Implementation out of date (load same agent)")
                for a_i in range(maddpg.nagents_team):
                    threads.append(mp.Process(target=updates.update_thread,args=(a_i,config,env.team_replay_buffer,env.opp_replay_buffer,number_of_updates,update_session)))
            else:                    
                for a_i in range(1):
                    threads.append(mp.Process(target=updates.update_thread,args=(a_i,config,env.team_replay_buffer,env.opp_replay_buffer,number_of_updates,update_session)))
            print("Launching update")
            start = time.time()
            [thr.start() for thr in threads]
            [thr.join() for thr in threads]
            update_session +=1
            env.update_counter.copy_(torch.zeros(num_envs,requires_grad=False))


        for envs in exp_indices:
            for exp_i in envs:
                exp_i.copy_(torch.tensor(0,requires_grad=False))
        number_of_updates = int(env.update_counter.sum().item())
        env.update_counter.copy_(torch.zeros(num_envs,requires_grad=False))

        env.halt.copy_(torch.tensor(0,requires_grad=False).byte())
        env.ready.copy_(torch.zeros(num_envs,requires_grad=False).byte())

                            
        maddpg.load_same_ensembles(env.ensemble_path,0,maddpg.nagents_team,load_same_agent=config.load_same_agent)        


        training = (len(env.team_replay_buffer) >= config.batch_size)
        if training:
            threads = []
            pt_inject_proba = max(0.0, 1.0*config.pt_inject_anneal_ep - update_session) / (config.pt_inject_anneal_ep)
            pt_inject_proba = config.final_pt_inject_proba + (config.init_pt_inject_proba - config.final_pt_inject_proba) * pt_inject_proba
            PT = (np.random.uniform(0,1) < pt_inject_proba) and (config.pt or config.use_pt_data)
            print("Update Session:",update_session)
            if PT or config.use_pt_data:

                print("PT sample probability:",pt_inject_proba)
                print("Using PT sample:",PT)

        
            if PT:
                rb_i = np.random.randint(num_buffers)

                for a_i in range(1):
                    threads.append(mp.Process(target=updates.update_thread,args=(a_i,config,pt_trb,pt_orb,number_of_updates,update_session)))
            else:
                if update_session > 75:
                    for a_i in range(1):
                        threads.append(mp.Process(target=updates.update_thread,args=(a_i,config,env.team_replay_buffer,env.opp_replay_buffer,number_of_updates,update_session)))
                [thr.start() for thr in threads]
            print("Launching update")
            start = time.time()

            update_session +=1
            agentID = 0
            buffer_size = len(env.team_replay_buffer)

            number_of_updates = 250
            batches_to_sample = 50
            if len(env.team_replay_buffer) < batch_size*(batches_to_sample):
                batches_to_sample = 1

            priorities = [] 

            for ensemble in range(config.k_ensembles):
                for up in range(number_of_updates):
                    offset = up % batches_to_sample
                    m = 0


                    if not config.load_same_agent:
                        inds = env.team_replay_buffer.get_PER_inds(agentID,batch_size,ensemble)
                    else:
                        if up % batches_to_sample == 0:
                            if PT:
                                #inds = pt_trb.get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                                inds = np.random.choice(np.arange(len(pt_trb)), size=config.batch_size*batches_to_sample, replace=False)

                            else:
                                #inds = team_replay_buffer.get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                                inds = np.random.choice(np.arange(len(env.team_replay_buffer)), size=config.batch_size*batches_to_sample, replace=False)
                            #if len(priorities) > 1:
                            #    for c,p in enumerate(priorities):
                            #        if PT:
                            #            pt_trb.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                            #        else:
                            #            team_replay_buffer.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                            priorities = [] 

                    #inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)

                    # FOR THE LOVE OF GOD DONT USE TORCH TO GET INDICES

                    if config.lstm_crit:
                        team_sample = env.team_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                                    to_gpu=config.to_gpu, device=maddpg.torch_device)
                        opp_sample = env.opp_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                                    to_gpu=config.to_gpu, device=maddpg.torch_device)

                        if not load_same_agent:
                            print("No implementation")

                        else:
                            train_actor = (len(env.team_replay_buffer) > 10000) and False # and (update_session % 2 == 0)
                            train_critic = (len(env.team_replay_buffer) > config.batch_size)
                            if train_critic:
                                priorities.append(maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=config.forward_pass,load_same_agent=config.load_same_agent,
                                                critic=True,policy=False,session_path=config.session_path,lstm_burn_in=config.burn_in_lstm))
                            if train_actor: # only update actor once 1 mill
                                _ = maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=config.forward_pass,
                                                                            load_same_agent=config.load_same_agent,critic=False,policy=True,session_path=config.session_path)

                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                if train_critic and not train_actor:
                                    maddpg.update_agent_critic(0,number_of_updates/10)
                                elif train_critic and train_actor:
                                    maddpg.update_agent_targets(0,number_of_updates/10)
                    else:
                        if PT:
                            
                            team_sample = pt_trb.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                        to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                            opp_sample = pt_orb.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                        to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                        else:
                            team_sample = env.team_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                        to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                            opp_sample = env.opp_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                        to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                        if not config.load_same_agent:
                            print("no implementation")
                        else:
                            train_actor = (len(env.team_replay_buffer) > 1000) and False # and (update_session % 2 == 0)
                            train_critic = (len(env.team_replay_buffer) > 1000)
                            if train_critic:
                                priorities.append(maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',
                                                                                    forward_pass=config.forward_pass,load_same_agent=config.load_same_agent,
                                                                                    critic=True,policy=False,session_path=config.session_path))
                            if train_actor: # only update actor once 1 mill
                                _ = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=config.forward_pass,
                                                                    load_same_agent=config.load_same_agent,
                                                                    critic=False,policy=True,session_path=config.session_path)

                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                if train_critic and not train_actor:
                                    maddpg.update_agent_critic(0,number_of_updates/10)
                                elif train_critic and train_actor:
                                    maddpg.update_agent_targets(0,number_of_updates/10)
                    if config.sil:
                        for i in range(config.update_ratio):
                            inds = team_replay_buffer.get_SIL_inds(agentID=m,batch_size=config.batch_size)
                            team_sample = team_replay_buffer.sample(inds,
                                                    to_gpu=config.to_gpu,norm_rews=False)
                            opp_sample = opp_replay_buffer.sample(inds,
                                                    to_gpu=config.to_gpu,norm_rews=False)
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
                if not config.load_same_agent:
                    maddpg.update_agent_targets(agentID,number_of_updates)
                    maddpg.save_agent(config.load_path,update_session,agentID,torch_device=maddpg.torch_device)
                    maddpg.save_ensemble(config.ensemble_path,ensemble,agentID,torch_device=maddpg.torch_device)
                else:
                    maddpg.update_agent_targets(0,number_of_updates/10)
                    print(time.time()-start,"<-- Critic Update Cycle")

                    [thr.join() for thr in threads]
                    maddpg.load_ensemble_policy(ensemble_path,ensemble,0) # load only policy from updated policy thread
                    [maddpg.save_agent(load_path,update_session,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(config.num_left)]
                    [maddpg.save_ensemble(config.ensemble_path,ensemble,i,config.load_same_agent,torch_device=maddpg.torch_device) for i in range(config.num_left)]
                print(time.time()-start,"<-- Full Cycle")
                cycle += 1


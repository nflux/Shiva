from maddpg import MADDPG

# updates policy only
def update_thread(agentID,to_gpu,buffer_size,batch_size,team_replay_buffer,opp_replay_buffer,number_of_updates,
                            load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_policy,k_ensembles,SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu,session_path,data_parallel,lstm_burn_in):
    start = time.time()
    if len(team_replay_buffer) > 1000:
        
        initial_models = [ensemble_path + ("ensemble_agent_%i/model_%i.pth" % (i,0)) for i in range(num_TA)]
        #maddpg = dill.loads(maddpg_pick)
        maddpg = MADDPG.init_from_save_evaluation(initial_models,num_TA) # from evaluation method just loads the networks
        if multi_gpu:
            maddpg.torch_device = torch.device("cuda:3")
        if to_gpu:
            maddpg.device = 'cuda'
        number_of_updates = 250
        batches_to_sample = 50

        if len(team_replay_buffer) < batch_size*(batches_to_sample):
            batches_to_sample = 1
        for ensemble in range(k_ensembles):

            maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
            maddpg.load_same_ensembles(ensemble_path,ensemble,maddpg.nagents_team,load_same_agent=load_same_agent)
            

            #start = time.time()
            #for up in range(int(np.floor(number_of_updates/k_ensembles))):
            for up in range(number_of_updates):
                m = 0
                #m = np.random.randint(num_TA)

                # if not load_same_agent:
                #     inds = team_replay_buffer.get_PER_inds(agentID,batch_size,ensemble)
                # else:
                #     inds = team_replay_buffer.get_PER_inds(m,batch_size,ensemble)

                # does not care about priority
                offset = up % batches_to_sample
                if up % batches_to_sample == 0:
                    inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size*batches_to_sample, replace=False)
                if LSTM_policy or LSTM:
                    team_sample = team_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                                to_gpu=to_gpu,device=maddpg.torch_device)
                    opp_sample = opp_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                                to_gpu=to_gpu,device=maddpg.torch_device)

                    if not load_same_agent:
                        print("No implementation")
                    else:                        
                        _ = maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,critic=False,policy=True,session_path=session_path,lstm_burn_in=lstm_burn_in)
                        if up % number_of_updates/10 == 0: # update target half way through
                            maddpg.update_agent_actor(0,number_of_updates/10)
                else:
                    team_sample = team_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                    opp_sample = opp_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                    if not load_same_agent:
                        print("No implementation")
                    else:
                        _ = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,critic=False,policy=True,session_path=session_path)
                        #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                        if up % number_of_updates/10 == 0: # update target half way through
                            maddpg.update_agent_actor(0,number_of_updates/10)
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
                maddpg.save_agent(load_path,update_session,agentID,torch_device=maddpg.torch_device)
                maddpg.save_ensemble(ensemble_path,ensemble,agentID,torch_device=maddpg.torch_device)
            else:
                [maddpg.save_agent(load_path,update_session,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
                [maddpg.save_ensemble(ensemble_path,ensemble,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
    print(time.time()-start,"<-- Policy Update Cycle")

def imitation_thread(agentID,to_gpu,buffer_size,batch_size,team_replay_buffer,opp_replay_buffer,number_of_updates,
                     load_path,update_session,ensemble_path,forward_pass,LSTM,LSTM_policy,seq_length,k_ensembles,
                     SIL,SIL_update_ratio,num_TA,load_same_agent,multi_gpu,session_path,data_parallel,lstm_burn_in):
    start = time.time()
    initial_models = [ensemble_path + ("ensemble_agent_%i/model_%i.pth" % (i,0)) for i in range(num_TA)]
    #maddpg = dill.loads(maddpg_pick)
    maddpg = MADDPG.init_from_save_evaluation(initial_models,num_TA) # from evaluation method just loads the networks

    number_of_updates = 6000
    batches_to_sample = 50
    if len(team_replay_buffer) < batch_size*(batches_to_sample):
        batches_to_sample = 1
    for ensemble in range(k_ensembles):
        if multi_gpu:
            maddpg.torch_device = torch.device("cuda:3")
        if to_gpu:
            maddpg.device = 'cuda'
        maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
        maddpg.load_same_ensembles(ensemble_path,ensemble,maddpg.nagents_team,load_same_agent=load_same_agent)
        m = 0
        #start = time.time()
        #for up in range(int(np.floor(number_of_updates/k_ensembles))):
        for up in range(number_of_updates):
            offset = up % batches_to_sample
            if up % batches_to_sample == 0:
                inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size*batches_to_sample, replace=False)
            if LSTM_policy:
                team_sample = team_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                            to_gpu=to_gpu,device=maddpg.torch_device)
                opp_sample = opp_replay_buffer.sample_LSTM(inds[batch_size*offset:batch_size*(offset+1)],
                                                                to_gpu=to_gpu,device=maddpg.torch_device)
               
                if not load_same_agent:
                    print("no implementation")
                else:
                    maddpg.pretrain_actor_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',load_same_agent=load_same_agent,lstm_burn_in=lstm_burn_in)
                    if up % number_of_updates/10 == 0: # update target half way through
                        maddpg.update_agent_actor(0,number_of_updates/10)

            else:
                team_sample = team_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                            to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                opp_sample = opp_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                            to_gpu=to_gpu,norm_rews=False,device=maddpg.torch_device)
                if not load_same_agent:
                    print("implementation missing")
                else:
                    maddpg.pretrain_actor(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,session_path=session_path)
                    #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                    if up % number_of_updates/10 == 0: # update target half way through
                        maddpg.update_agent_actor(0,number_of_updates/10)

        #print(time.time()-start)
        if not load_same_agent:
            maddpg.update_agent_targets(agentID,number_of_updates)
            maddpg.save_agent(load_path,update_session,agentID)
            maddpg.save_ensemble(ensemble_path,ensemble,agentID)
        else:
            [maddpg.save_agent(load_path,update_session,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
            [maddpg.save_ensemble(ensemble_path,ensemble,i,load_same_agent,torch_device=maddpg.torch_device) for i in range(num_TA)]
    print(time.time()-start,"<-- Policy Update Cycle")
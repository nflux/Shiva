from .maddpg import MADDPG

class Update:
    def __init__(self,config,team_replay_buffer,opp_replay_buffer):
        self.config = config
        self.team_replay_buffer = team_replay_buffer
        self.opp_replay_buffer = opp_replay_buffer

    def main_update(self):
        
        iterations_per_push = 1
        update_session = 0

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
                        threads.append(mp.Process(target=update.update_thread,args=(a_i,number_of_updates,update_session)))
                else:                    
                    for a_i in range(1):
                        threads.append(mp.Process(target=updates.update_thread,args=(a_i,number_of_updates,update_session)))
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

    # updates policy only
    def update_thread(self,agentID,number_of_updates,update_session):
        start = time.time()
        config = self.config
        team_replay_buffer = self.team_replay_buffer
        opp_replay_buffer = self.opp_replay_buffer
        if len(team_replay_buffer) > 1000:
            
            initial_models = [config.ensemble_path + ("ensemble_agent_%i/model_%i.pth" % (i,0)) for i in range(config.num_left)]
            #maddpg = dill.loads(maddpg_pick)
            maddpg = MADDPG.init_from_save_evaluation(initial_models,config.num_left) # from evaluation method just loads the networks
            if config.multi_gpu:
                maddpg.torch_device = torch.device("cuda:3")
            if config.to_gpu:
                maddpg.device = 'cuda'
            number_of_updates = 250
            batches_to_sample = 50

            if len(team_replay_buffer) < config.batch_size*(batches_to_sample):
                batches_to_sample = 1
            for ensemble in range(config.k_ensembles):

                maddpg.prep_training(device=maddpg.device,torch_device=maddpg.torch_device)
                maddpg.load_same_ensembles(config.ensemble,ensemble,maddpg.nagents_team,load_same_agent=config.load_same_agent)
                

                #start = time.time()
                #for up in range(int(np.floor(number_of_updates/k_ensembles))):
                for up in range(number_of_updates):
                    m = 0
                    #m = np.random.randint(num_TA)

                    # if not load_same_agent:
                    #     inds = team_replay_buffer.get_PER_inds(agentID,config.batch_size,ensemble)
                    # else:
                    #     inds = team_replay_buffer.get_PER_inds(m,config.batch_size,ensemble)

                    # does not care about priority
                    offset = up % batches_to_sample
                    if up % batches_to_sample == 0:
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=config.batch_size*batches_to_sample, replace=False)
                    if config.lstm_pol or config.lstm_crit:
                        team_sample = team_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                                    to_gpu=config.to_gpu,device=maddpg.torch_device)
                        opp_sample = opp_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                                    to_gpu=config.to_gpu,device=maddpg.torch_device)

                        if not config.load_same_agent:
                            print("No implementation")
                        else:                        
                            _ = maddpg.update_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=config.forward_pass,
                                                    load_same_agent=config.load_same_agent,critic=False,policy=True,session_path=config.session_path,lstm_burn_in=config.burn_in_lstm)
                            if up % number_of_updates/10 == 0: # update target half way through
                                maddpg.update_agent_actor(0,number_of_updates/10)
                    else:
                        team_sample = team_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                    to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                        opp_sample = opp_replay_buffer.sample(inds[config.batch_size*offset:config.batch_size*(offset+1)],
                                                    to_gpu=config.to_gpu,norm_rews=False,device=maddpg.torch_device)
                        if not config.load_same_agent:
                            print("No implementation")
                        else:
                            _ = maddpg.update_centralized_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=config.forward_pass,
                                                                load_same_agent=config.load_same_agent,critic=False,policy=True,session_path=config.session_path)
                            #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                            if up % number_of_updates/10 == 0: # update target half way through
                                maddpg.update_agent_actor(0,number_of_updates/10)
                    if config.sil:
                        for i in range(config.update_ratio):
                            inds = team_replay_buffer.get_SIL_inds(agentID=m,batch_size=config.batch_size)
                            team_sample = team_replay_buffer.sample(inds,
                                                    to_gpu=config.to_gpu,norm_rews=False)
                            opp_sample = opp_replay_buffer.sample(inds,
                                                    to_gpu=config.to_gpu,norm_rews=False)
                            priorities = maddpg.SIL_update(team_sample, opp_sample, agentID, 'team') # 
                            team_replay_buffer.update_SIL_priorities(agentID=m,inds = inds, prio=priorities)

                #print(time.time()-start)
                if not config.load_same_agent:
                    maddpg.update_agent_targets(agentID,number_of_updates)
                    maddpg.save_agent(config.load_path,update_session,agentID,torch_device=maddpg.torch_device)
                    maddpg.save_ensemble(config.ensemble,ensemble,agentID,torch_device=maddpg.torch_device)
                else:
                    [maddpg.save_agent(config.load_path,update_session,i,config.load_same_agent,torch_device=maddpg.torch_device) for i in range(config.num_left)]
                    [maddpg.save_ensemble(config.ensemble_path,ensemble,i,config.load_same_agent,torch_device=maddpg.torch_device) for i in range(config.num_left)]
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
                team_sample = team_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
                                                            to_gpu=to_gpu,device=maddpg.torch_device)
                opp_sample = opp_replay_buffer.sample(inds[batch_size*offset:batch_size*(offset+1)],
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
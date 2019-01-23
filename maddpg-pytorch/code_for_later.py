
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
            
            
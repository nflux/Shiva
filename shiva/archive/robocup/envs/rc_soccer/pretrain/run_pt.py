import os
from algorithms.maddpg import MADDPG
from utils.misc import pretrain_process
from rc_env import rc_env
from utils.tensor_buffer import ReplayTensorBuffer
import numpy as np
import collections
import torch

def pretrain(maddpg, env, team_replay_buffer, opp_replay_buffer, num_TA = 3, num_OA = 3, critic_mod_both=True, device='cuda', Imitation_exploration=False, episode_length=500, num_features=0):
    pt_critic_updates = 30000
    pt_actor_updates = 30000
    pt_actor_critic_updates = 0
    pt_imagination_branch_pol_updates = 100
    pt_episodes = 1000# num of episodes that you observed in the gameplay between npcs
    pt_timesteps = 125000# number of timesteps to load in from files
    pt_EM_updates = 300
    pt_beta = 1.0

    initial_beta = 0.3
    n_steps = 5
    gamma = 0.99 # discount
    exps = None
    SIL = False

    # -------------------------------------
    # PRETRAIN ############################
    if Imitation_exploration:
        left_files = []
        right_files = []
        # team_files = ['Pretrain_Files/3v3_CentQ/base_left-11.log','Pretrain_Files/3v3_CentQ/base_left-7.log','Pretrain_Files/3v3_CentQ/base_left-8.log','Pretrain_Files/3v3_CentQ/base_right-2.log','Pretrain_Files/3v3_CentQ/base_right-3.log','Pretrain_Files/3v3_CentQ/base_right-4.log']
        # opp_files = ['Pretrain_Files/base_left-1.log','Pretrain_Files/base_left-2.log']
        if os.path.isdir(os.getcwd() + '/pt_logs'):
            team_files = os.listdir(os.getcwd() + '/pt_logs')
            left_files = [os.getcwd() + '/pt_logs/' + f for f in team_files if '_left_' in f]
            right_files = [os.getcwd() + '/pt_logs/' + f for f in team_files if '_right_' in f]
        else:
            print('log directory DNE')
            exit(0)

        team_pt_status, team_pt_obs,team_pt_actions, opp_pt_status, opp_pt_obs, opp_pt_actions, status = pretrain_process(left_fnames=left_files, right_fnames=right_files, num_features = num_features)

        # Count up everything besides IN_GAME to get number of episodes
        collect = collections.Counter(status[0])
        pt_episodes = collect[1] + collect[2] + collect[3] + collect[4]
        pt_time_step = 0

        ################## Base Left #########################
        for ep_i in range(pt_episodes):
            # if ep_i % 100 == 0:
            #     print("Pushing Pretrain Base-Left Episode:",ep_i)
            
            team_n_step_rewards = []
            team_n_step_obs = []
            team_n_step_acs = []
            team_n_step_next_obs = []
            team_n_step_dones = []
            team_n_step_ws = []

            opp_n_step_rewards = []
            opp_n_step_obs = []
            opp_n_step_acs = []
            opp_n_step_next_obs = []
            opp_n_step_dones = []
            opp_n_step_ws = []

            #define/update the noise used for exploration
            explr_pct_remaining = 0.0
            beta_pct_remaining = 0.0
            maddpg.scale_noise(0.0)
            maddpg.reset_noise()
            maddpg.scale_beta(pt_beta)
            d = 0
            
            for et_i in range(episode_length):
                world_stat = status[0][pt_time_step]
                d = 0
                if world_stat != 0.0:
                    print('This is called done!')
                    d = 1
                # print('actions len', len(team_pt_actions))
                # print('timestep', pt_time_step, 'actions', team_pt_actions[pt_time_step].shape)
                team_n_step_acs.append(team_pt_actions[pt_time_step])
                team_n_step_obs.append(team_pt_obs[pt_time_step])
                team_n_step_ws.append(world_stat)
                team_n_step_next_obs.append(team_pt_obs[pt_time_step+1])
                team_n_step_rewards.append(np.hstack([env.getPretrainRew(world_stat,d,"base_left") for i in range(num_TA)]))          
                team_n_step_dones.append(d)

                opp_n_step_acs.append(opp_pt_actions[pt_time_step])
                opp_n_step_obs.append(opp_pt_obs[pt_time_step])
                opp_n_step_ws.append(world_stat)
                opp_n_step_next_obs.append(opp_pt_obs[pt_time_step+1])
                opp_n_step_rewards.append(np.hstack([env.getPretrainRew(world_stat,d,"base_right") for i in range(num_TA)]))          
                opp_n_step_dones.append(d)

                # Store variables for calculation of MC and n-step targets for team
                pt_time_step += 1
                if d == 1: # Episode done
                    n_step_gammas = np.array([[gamma**step for a in range(num_TA)] for step in range(n_steps)])
                    # NOTE: Assume M vs M and critic_mod_both == True
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
                            # priorities = np.array([np.zeros(k_ensembles) for i in range(num_TA)])
                            # priorities[:,current_ensembles] = 5.0
                            # print(current_ensembles)
                            if SIL:
                                SIL_priorities = np.ones(num_TA)*default_prio
                            # print('Got here')
                            # print(team_n_step_obs[n].shape)
                            # print(team_n_step_acs[n].shape)
                            # print(n_step_next_ob_team.shape)
                            # print(n_step_done_team)
                            # print(team_all_MC_targets[et_i-n].shape)
                            # print(n_step_targets_team)
                            # print(team_n_step_ws[n])

                            # exit(0)
                            exp_team = np.column_stack((team_n_step_obs[n],
                                                team_n_step_acs[n],
                                                np.expand_dims(team_n_step_rewards[n], 1),
                                                n_step_next_ob_team,
                                                np.expand_dims([n_step_done_team for i in range(num_TA)], 1),
                                                np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                                np.expand_dims(n_step_targets_team, 1),
                                                np.expand_dims([team_n_step_ws[n] for i in range(num_TA)], 1),
                                                # priorities,
                                                np.expand_dims([default_prio for i in range(num_TA)],1)))


                            exp_opp = np.column_stack((opp_n_step_obs[n],
                                                opp_n_step_acs[n],
                                                np.expand_dims(opp_n_step_rewards[n], 1),
                                                n_step_next_ob_opp,
                                                np.expand_dims([n_step_done_opp for i in range(num_OA)], 1),
                                                np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                                np.expand_dims(n_step_targets_opp, 1),
                                                np.expand_dims([opp_n_step_ws[n] for i in range(num_OA)], 1),
                                                # priorities,
                                                np.expand_dims([default_prio for i in range(num_TA)],1)))
                    
                            exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)

                            if exps is None:
                                exps = torch.from_numpy(exp_comb)
                            else:
                                exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)
                            
                        del exps
                        exps = None
                    break


                # if d == 1: # Episode done
                #     # Calculate n-step and MC targets
                #     for n in range(et_i+1):
                #         MC_targets = []
                #         n_step_targets = []
                #         for a in range(env.num_TA):
                #             MC_target = 0
                #             n_step_target = 0
                            
                #             for step in range(et_i+1 - n): # sum MC target
                #                 MC_target += team_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
                #             MC_targets.append(MC_target)
                #             if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
                #                 for step in range(n_steps): 
                #                     n_step_target += team_n_step_rewards[n + step][a] * gamma**(step)
                #                 n_step_targets.append(n_step_target)
                #                 n_step_next_ob = n_step_next_obs[n - 1 + n_steps]
                #                 n_step_done = team_n_step_dones[n - 1 + n_steps]
                #             else: # n-step = MC if less than n steps remaining
                #                 n_step_target = MC_target
                #                 n_step_targets.append(n_step_target)
                #                 n_step_next_ob = n_step_next_obs[-1]
                #                 n_step_done = team_n_step_dones[-1]
                #         # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
                #         #pretrain_buffer.push
                #         #team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                #         #                        [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                #         #                        [team_n_step_ws[n] for i in range(env.num_TA)])
                #         team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
                #                                 [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
                #                                 [team_n_step_ws[n] for i in range(env.num_TA)])

                #     # pt_time_step +=1
                #     break
        
        del team_pt_obs
        del team_pt_status
        del team_pt_actions

        del opp_pt_obs
        del opp_pt_status
        del opp_pt_actions
        ################## Base Right ########################

        # pt_time_step = 0
        # for ep_i in range(0, pt_episodes):
        #     if ep_i % 100 == 0:
        #         print("Pushing Pretrain Base Right Episode:",ep_i)
                
                
        #     # team n-step
        #     team_n_step_rewards = []
        #     team_n_step_obs = []
        #     team_n_step_acs = []
        #     n_step_next_obs = []
        #     team_n_step_dones = []
        #     team_n_step_ws = []

        #     #define/update the noise used for exploration
        #     explr_pct_remaining = 0.0
        #     beta_pct_remaining = 0.0
        #     maddpg.scale_noise(0.0)
        #     maddpg.reset_noise()
        #     maddpg.scale_beta(pt_beta)
        #     d = False
            
        #     for et_i in range(0, episode_length):            
        #         world_stat = opp_pt_status[pt_time_step]
        #         d = False
        #         if world_stat != 0.0:
        #             d = True

        #         #### Team ####
        #         team_n_step_acs.append(opp_pt_actions[pt_time_step])
        #         team_n_step_obs.append(np.array(opp_pt_obs[pt_time_step]).T)
        #         team_n_step_ws.append(world_stat)
        #         n_step_next_obs.append(np.array(opp_pt_obs[pt_time_step+1]).T)
        #         team_n_step_rewards.append(np.hstack([env.getPretrainRew(world_stat,d,"base_left") for i in range(env.num_TA) ]))          
        #         team_n_step_dones.append(d)

        #         # Store variables for calculation of MC and n-step targets for team
        #         pt_time_step += 1
        #         if d == True: # Episode done
        #             # Calculate n-step and MC targets
        #             for n in range(et_i+1):
        #                 MC_targets = []
        #                 n_step_targets = []
        #                 for a in range(env.num_TA):
        #                     MC_target = 0
        #                     n_step_target = 0
                            
        #                     for step in range(et_i+1 - n): # sum MC target
        #                         MC_target += team_n_step_rewards[et_i - step][a] * gamma**(et_i - n - step)
        #                     MC_targets.append(MC_target)
        #                     if (et_i + 1) - n >= n_steps: # sum n-step target (when more than n-steps remaining)
        #                         for step in range(n_steps): 
        #                             n_step_target += team_n_step_rewards[n + step][a] * gamma**(step)
        #                         n_step_targets.append(n_step_target)
        #                         n_step_next_ob = n_step_next_obs[n - 1 + n_steps]
        #                         n_step_done = team_n_step_dones[n - 1 + n_steps]
        #                     else: # n-step = MC if less than n steps remaining
        #                         n_step_target = MC_target
        #                         n_step_targets.append(n_step_target)
        #                         n_step_next_ob = n_step_next_obs[-1]
        #                         n_step_done = team_n_step_dones[-1]
        #                 # obs, acs, immediate rewards, next_obs, dones, mc target, n-step target
        #                 #pretrain_buffer.push
        #                 #team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
        #                 #                        [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
        #                 #                        [team_n_step_ws[n] for i in range(env.num_TA)])
        #                 team_replay_buffer.push(team_n_step_obs[n], team_n_step_acs[n],team_n_step_rewards[n],n_step_next_ob,
        #                                         [n_step_done for i in range(env.num_TA)],MC_targets, n_step_targets,
        #                                         [team_n_step_ws[n] for i in range(env.num_TA)])

        #             pt_time_step +=1
        #             break
                    

        
                    ##################################################
        print('Victory!!!!')
        exit(0)
        maddpg.prep_training(device=device)
        
        # if I2A:
        #     # pretrain EM
        #     for i in range(pt_EM_updates):
        #         if i%100 == 0:
        #             print("Petrain EM update:",i)
        #         for u_i in range(1):
        #             for a_i in range(env.num_TA):
        #                 inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #                 #sample = pretrain_buffer.sample(batch_size,
        #                 sample = team_replay_buffer.sample(inds,
        #                                             to_gpu=to_gpu,norm_rews=False, device=device)
        #                 maddpg.update_EM(sample, a_i,'team')
        #             maddpg.niter+=1
                        
        #     if Imitation_exploration:
        #         # pretrain policy prime
        #             # pretrain critic
        #         for i in range(pt_actor_updates):
        #             if i%100 == 0:
        #                 print("Petrain Prime update:",i)
        #             for u_i in range(1):
        #                 for a_i in range(env.num_TA):
        #                     inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #                     sample = team_replay_buffer.sample(inds,
        #                                                     to_gpu=to_gpu,norm_rews=False, device=device)
        #                     maddpg.pretrain_prime(sample, a_i,'team')
        #                 maddpg.niter +=1
        #         if imagination_policy_branch and I2A:
        #             for i in range(pt_imagination_branch_pol_updates):
        #                 if i%100 == 0:
        #                     print("Petrain imag policy update:",i)
        #                 for u_i in range(1):
        #                     for a_i in range(env.num_TA):
        #                         inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #                         sample = team_replay_buffer.sample(inds,
        #                                                     to_gpu=to_gpu,norm_rews=False, device=device)
        #                         maddpg.pretrain_imagination_policy(sample, a_i,'team')
        #                     maddpg.niter +=1

                    

        # # pretrain policy
        # for i in range(pt_actor_updates):
        #     if i%100 == 0:
        #         print("Petrain actor update:",i)
        #     for u_i in range(1):
        #         for a_i in range(env.num_TA):
        #             inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #             sample = team_replay_buffer.sample(inds,
        #                                         to_gpu=to_gpu,norm_rews=False, device=device)
        #             maddpg.pretrain_actor(sample, a_i,'team')
        #         maddpg.niter +=1

        # maddpg.update_hard_policy()
        
        
        
        # if not critic_mod: # non-centralized Q
        #     # pretrain critic
        #     for i in range(pt_critic_updates):
        #         if i%100 == 0:
        #             print("Petrain critic update:",i)
        #         for u_i in range(1):
        #             for a_i in range(env.num_TA):
        #                 inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #                 sample = team_replay_buffer.sample(inds,
        #                                                 to_gpu=to_gpu,norm_rews=False)
        #                 maddpg.pretrain_critic(sample, a_i,'team')
        #             maddpg.niter +=1
        #     maddpg.update_hard_critic()


        #     maddpg.scale_beta(initial_beta) # 
        #     # pretrain true actor-critic (non-imitation) + policy prime
        #     for i in range(pt_actor_critic_updates):
        #         if i%100 == 0:
        #             print("Petrain critic/actor update:",i)
        #         for u_i in range(1):
        #             for a_i in range(env.num_TA):
        #                 inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
        #                 sample = team_replay_buffer.sample(inds,
        #                                             to_gpu=to_gpu,norm_rews=False)
        #                 maddpg.update(sample, a_i, 'team' )
        #                 if SIL:
        #                     for i in range(SIL_update_ratio):
        #                         team_sample,inds = team_replay_buffer.sample_SIL(agentID=a_i,batch_size=batch_size,
        #                                             to_gpu=to_gpu,norm_rews=False)
        #                         priorities = maddpg.SIL_update(team_sample, opp_sample, a_i, 'team', 
        #                                         centQ=critic_mod) # 
        #                         team_replay_buffer.update_priorities(agentID=a_i,inds = inds, prio=priorities)
        #             maddpg.update_all_targets()
        if False: # centralized Q
            for i in range(pt_critic_updates):
                if i%100 == 0:
                    print("Petrain critic update:",i)
                for u_i in range(1):
                    for a_i in range(env.num_TA):
                        inds = np.random.choice(np.arange(len(team_replay_buffer)), size=batch_size, replace=False)
                        team_sample = team_replay_buffer.sample(inds,
                                                                to_gpu=to_gpu,norm_rews=False, device=device)
                        opp_sample = opp_replay_buffer.sample(inds,
                                                            to_gpu=to_gpu,norm_rews=False, device=device)
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

if __name__ == "__main__":
    USE_CUDA = False
    if USE_CUDA:
        device = 'cuda'
        to_gpu = True
    else:
        to_gpu = False
        device = 'cpu'
    
    n_training_threads = 4
    batch_size = 128
    tau = 0.001

    num_TA = 3
    num_OA = 3
    goalie = False
    seed = 123
    feature_level = 'low'
    action_level = 'low'
    untouched_time = 200
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

    log_dir = 'nan_log'
    rcss_log_game = False #Logs the game using rcssserver
    hfo_log_game = False #Logs the game using HFO
    # default settings ---------------------
    num_episodes = 10000000
    episode_length = 500 # FPS
    untouched_time = 500
    burn_in_iterations = 500 # for time step
    burn_in_episodes = float(burn_in_iterations)/untouched_time
    deterministic = True

    # --------------------------------------
    # hyperparams--------------------------
    batch_size = 128
    hidden_dim = int(512)

    tau = 0.001 # soft update rate

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
    # --------------------------------------
    # LSTM -------------------------------------------
    LSTM = False # Critic only
    if LSTM:
        seq_length = 40 # Must be divisible by 2
    else:
        seq_length = 0
    if seq_length % 2 != 0:
        print('Seq length must be divisible by 2')
        exit(0)
    hidden_dim_lstm = 512

    if action_level == 'high':
        discrete_action = True
    else:
        discrete_action = False
    if not USE_CUDA:
        torch.set_num_threads(n_training_threads)
    # -------------------------------------------------

    # dummy env that isn't used explicitly ergo used for dimensions andd methods
    env = rc_env(num_TNPC = 0,num_TA=num_TA,num_OA=num_OA, num_ONPC=0, goalie=goalie,
                    num_trials = num_episodes, fpt = episode_length, seed=seed, # create environment
                    feat_lvl = feature_level, act_lvl = action_level, untouched_time = untouched_time,fullstate=True,
                    ball_x_min=ball_x_min, ball_x_max=ball_x_max, ball_y_min=ball_y_min, ball_y_max=ball_y_max,
                    offense_on_ball=False,port=65000,log_dir=log_dir, rcss_log_game=rcss_log_game, hfo_log_game=hfo_log_game,
                    agents_x_min=agents_x_min, agents_x_max=agents_x_max, agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                    change_every_x=change_every_x, change_agents_x=change_agents_x, change_agents_y=change_agents_y,
                    change_balls_x=change_balls_x, change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=True,
                    defense_team_bin='base', offense_team_bin='base', run_server=False, deterministic=True)
    
    obs_dim_TA = env.team_num_features
    obs_dim_OA = env.opp_num_features
    acs_dim = 8
    replay_memory_size = 300000
    k_ensembles = 1
    # num_features = 59 + 13*(left_side-1) + 12*right_side + 4 + 1 + 2 + 1

    team_replay_buffer = ReplayTensorBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM,k_ensembles,SIL)

    opp_replay_buffer = ReplayTensorBuffer(replay_memory_size , num_TA,
                                        obs_dim_TA,acs_dim,batch_size, LSTM,k_ensembles,SIL)

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
                                LSTM=LSTM, seq_length=seq_length, hidden_dim_lstm=hidden_dim_lstm,only_policy=False,multi_gpu=multi_gpu)

    pretrain(maddpg, env, team_replay_buffer, opp_replay_buffer, num_TA=num_TA, num_OA=num_OA, critic_mod_both=critic_mod_both, device=device, Imitation_exploration=True, episode_length=500, num_features=obs_dim_TA)
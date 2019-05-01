import envs.rc_soccer.rc_env as rc
import utils.buffers as buff
import utils.misc as misc
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp
import os, dill
import pandas as pd
import algorithms.maddpg as mad_algo
import algorithms.updates as updates
from .pretrain import pretrain_process as pretrainer
import time
import numpy as np
import config as conf

class RoboEnvs:
    def __init__(self, config):
        self.config = config
        self.template_env = rc.rc_env(config, 0)
        self.obs_dim = self.template_env.team_num_features
        # self.maddpg = MADDPG.init(config, self.env)

        self.prox_item_size = config.num_left*(2*self.obs_dim + 2*config.ac_dim)
        self.team_replay_buffer = buff.init_buffer(config, config.lstm_crit or config.lstm_pol,
                                                    self.obs_dim, self.prox_item_size)

        self.opp_replay_buffer = buff.init_buffer(config, config.lstm_crit or config.lstm_pol,
                                                    self.obs_dim, self.prox_item_size)
        self.max_episodes_shared = 30
        self.total_dim = (self.obs_dim + config.ac_dim + 5) + config.k_ensembles + 1 + (config.hidden_dim_lstm*4) + self.prox_item_size

        self.shared_exps = [[torch.zeros(config.max_num_exps,2*config.num_left,self.total_dim,requires_grad=False).share_memory_() for _ in range(self.max_episodes_shared)] for _ in range(config.num_envs)]
        self.exp_indices = [[torch.tensor(0,requires_grad=False).share_memory_() for _ in range(self.max_episodes_shared)] for _ in range(config.num_envs)]
        self.ep_num = torch.zeros(config.num_envs,requires_grad=False).share_memory_()

        self.halt = Variable(torch.tensor(0).byte()).share_memory_()
        self.ready = torch.zeros(config.num_envs,requires_grad=False).byte().share_memory_()
        self.update_counter = torch.zeros(config.num_envs,requires_grad=False).share_memory_()

    def run(self):
        processes = []
        envs = []
        for i in range(self.config.num_envs):
            envs.append(rc.rc_env(self.config, self.config.port + (i * 1000)))
            processes.append(mp.Process(target=run_env, args=(dill.dumps(envs[i]),self.shared_exps[i],
                                        self.exp_indices[i],i,self.ready,self.halt,self.update_counter,
                                        (self.config.history+str(i)),self.ep_num,self.obs_dim)))

        for p in processes: # Starts environments
            p.start()

class RoboEnvsWrapper:
    def __init__(self, config_parse):
        self.config = conf.RoboConfig(config_parse)
        self.envs = RoboEnvs(self.config)
        self.maddpg = mad_algo.init_from_env(self.config, self.envs.template_env)
        self.update = updates.Update(self.config, self.envs.team_replay_buffer, self.envs.opp_replay_buffer)
        #Pretraining **Needs create_pretrain_files.py to test, importing HFO issue
        # self.pretrainer = pretrainer.pretrain(self.config, self.envs)

    def run(self):
        mp.set_start_method('forkserver',force=True)
        # self.pretrainer.pretraining()
        # self.maddpg.save_agent2d(config.load_path,0,config.load_same_agent,maddpg.torch_device)
        # [self.maddpg.save_ensemble(config.ensemble_path,0,i,config.load_same_agent,maddpg.torch_device) for i in range(config.num_left)] # Save agent2d into ensembles
        self.envs.run()
        self.maddpg.scale_beta(self.config.init_beta)
        self.update.main_update(self.envs, self.maddpg)


def run_env(env,shared_exps,exp_i,env_num,ready,halt,num_updates,history,ep_num,obs_dim):

    env = dill.loads(env)

    config = env.config
    if config.record_lib or config.record_serv:
        if os.path.isdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(env.port)):
            file_list = os.listdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(env.port))
            [os.remove(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(env.port) + '/' + f) for f in file_list]
        else:
            os.mkdir(os.getcwd() + '/pretrain/pretrain_data//pt_logs_' + str(env.port))

    if config.load_nets:
        maddpg = mad_algo.MADDPG.init_from_save_evaluation(config.initial_models,env.num_TA) # from evaluation method just loads the networks
    else:
        maddpg = mad_algo.init(config, env)        
        
    if config.to_gpu:
        maddpg.device = 'cuda'

    if config.multi_gpu:
        if env_num < 5:
            maddpg.torch_device = torch.device("cuda:1")
        else:
            maddpg.torch_device = torch.device("cuda:2")

    current_ensembles = config.current_ensembles
    preload_agent2d_path = ''

    maddpg.prep_training(device=maddpg.device,only_policy=False,torch_device=maddpg.torch_device)

    reward_total = [ ]
    num_steps_per_episode = []
    end_actions = [] 
    team_step_logger_df = pd.DataFrame()
    opp_step_logger_df = pd.DataFrame()

    prox_item_size = env.num_TA*(2*env.team_num_features + 2*env.acs_dim)
    exps = None
    t = 0

    # --------------------------------
    env.launch()
    if config.use_viewer:
        env._start_viewer()       

    time.sleep(3)
    for ep_i in range(0, config.num_ep):
        if config.to_gpu:
            maddpg.device = 'cuda'

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
        maddpg.prep_policy_rollout(device=maddpg.device,torch_device=maddpg.torch_device)


        
        #define/update the noise used for exploration
        if ep_i < config.burn_in_eps:
            explr_pct_remaining = 1.0
        else:
            explr_pct_remaining = max(0.0, 1.0*config.num_exp_eps - ep_i + config.burn_in_eps) / (config.num_exp_eps)
        beta_pct_remaining = max(0.0, 1.0*config.num_beta_eps - ep_i + config.burn_in_eps) / (config.num_beta_eps)
        
        # evaluation for 10 episodes every 100
        if ep_i % 10 == 0:
            maddpg.scale_noise(config.final_ou_noise_scale + (config.init_noise_scale - config.final_ou_noise_scale) * explr_pct_remaining)
        if ep_i % 100 == 0:
            maddpg.scale_noise(0.0)

        if config.lstm_crit:
            maddpg.zero_hidden(1,actual=True,target=True,torch_device=maddpg.torch_device)
        if config.lstm_pol:
            maddpg.zero_hidden_policy(1,maddpg.torch_device)
        maddpg.reset_noise()
        maddpg.scale_beta(config.final_beta + (config.init_beta - config.final_beta) * beta_pct_remaining)
        #for the duration of 100 episode with maximum length of 500 time steps
        time_step = 0
        team_kickable_counter = [0] * env.num_TA
        opp_kickable_counter = [0] * env.num_OA
        env.team_possession_counter = [0] * env.num_TA
        env.opp_possession_counter = [0] * env.num_OA
        #reducer = maddpg.team_agents[0].reducer

        # List of tensors sorted by proximity in terms of agents
        sortedByProxTeamList = []
        sortedByProxOppList = []
        for et_i in range(0, config.ep_length):

            if config.device == 'cuda':
                # gather all the observations into a torch tensor 
                torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False).cuda(non_blocking=True,device=maddpg.torch_device)
                            for i in range(env.num_TA)]

                # gather all the opponent observations into a torch tensor 
                torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False).cuda(non_blocking=True,device=maddpg.torch_device)
                            for i in range(env.num_OA)]
                

            else:
                # gather all the observations into a torch tensor 
                torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False)
                            for i in range(env.num_TA)]

                # gather all the opponent observations into a torch tensor 
                torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False)
                            for i in range(env.num_OA)] 
    
            # Get e-greedy decision
            if config.explore:
                team_randoms = misc.e_greedy_bool(env.num_TA,eps = (config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining),device=maddpg.torch_device)
                opp_randoms = misc.e_greedy_bool(env.num_OA,eps =(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining),device=maddpg.torch_device)
            else:
                team_randoms = misc.e_greedy_bool(env.num_TA,eps = 0,device=maddpg.torch_device)
                opp_randoms = misc.e_greedy_bool(env.num_OA,eps = 0,device=maddpg.torch_device)

            # get actions as torch Variables for both team and opp

            team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_opp,team_randoms,opp_randoms,parallel=False,explore=config.explore) # leave off or will gumbel sample
            # convert actions to numpy arrays

            team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
            #Converting actions to numpy arrays for opp agents
            opp_agent_actions = [ac.cpu().data.numpy() for ac in opp_torch_agent_actions]

            opp_params = np.asarray([ac[0][len(env.action_list):] for ac in opp_agent_actions])

            # this is returning one-hot-encoded action for each team agent
            team_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in team_agent_actions]])
            # this is returning one-hot-encoded action for each opp agent 
            opp_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in opp_agent_actions]])

            team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            opp_obs =  np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T
            
            # use random unif parameters if e_greedy
            team_noisey_actions_for_buffer = team_actions[0]
            team_params = np.array([val[0][len(env.action_list):] for val in team_agent_actions])
            opp_noisey_actions_for_buffer = opp_actions[0]
            opp_params = np.array([val[0][len(env.action_list):] for val in opp_agent_actions])

            team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes       
            team_actions_params_for_buffer = np.array([val[0] for val in team_agent_actions])
            opp_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in opp_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes

            opp_actions_params_for_buffer = np.array([val[0] for val in opp_agent_actions])

            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.team_kickable[i] for i in range(env.num_TA)])
            if kickable.any():
                team_kickable_counter = [tkc + 1 if kickable[i] else tkc for i,tkc in enumerate(team_kickable_counter)]
                
            # If kickable is True one of the teammate agents has possession of the ball
            kickable = np.array([env.opp_kickable[i] for i in range(env.num_OA)])
            if kickable.any():
                opp_kickable_counter = [okc + 1 if kickable[i] else okc for i,okc in enumerate(opp_kickable_counter)]
            
            team_possession_counter = [env.get_agent_possession_status(i, env.team_base) for i in range(env.num_TA)]
            opp_possession_counter = [env.get_agent_possession_status(i, env.opp_base) for i in range(env.num_OA)]

            sortedByProxTeamList.append(misc.constructProxmityList(env, team_obs.T, opp_obs.T, team_actions_params_for_buffer, opp_actions_params_for_buffer, env.num_TA, 'left'))
            sortedByProxOppList.append(misc.constructProxmityList(env, opp_obs.T, team_obs.T, opp_actions_params_for_buffer, team_actions_params_for_buffer, env.num_OA, 'right'))

            _,_,_,_,d,world_stat = env.Step(team_agents_actions, opp_agents_actions, team_params, opp_params,team_agent_actions,opp_agent_actions)

            team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])
            opp_rewards = np.hstack([env.Reward(i,'opp') for i in range(env.num_OA)])

            team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
            opp_next_obs = np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T

            
            team_done = env.d
            opp_done = env.d 

            team_n_step_rewards.append(team_rewards)
            team_n_step_obs.append(team_obs)
            team_n_step_next_obs.append(team_next_obs)
            team_n_step_acs.append(team_actions_params_for_buffer)
            team_n_step_dones.append(team_done)
            team_n_step_ws.append(world_stat)

            opp_n_step_rewards.append(opp_rewards)
            opp_n_step_obs.append(opp_obs)
            opp_n_step_next_obs.append(opp_next_obs)
            opp_n_step_acs.append(opp_actions_params_for_buffer)
            opp_n_step_dones.append(opp_done)
            opp_n_step_ws.append(world_stat)
            # ----------------------------------------------------------------
            # Reduce size of obs

            time_step += 1
            t += 1

            if t%3000 == 0:
                team_step_logger_df.to_csv(hist_dir + '/team_%s.csv' % history)
                opp_step_logger_df.to_csv(hist_dir + '/opp_%s.csv' % history)
                        
            team_episode = []
            opp_episode = []

            if d == 1 and et_i >= (config.seq_length-1): # Episode done 
                n_step_gammas = np.array([[config.gamma**step for a in range(env.num_TA)] for step in range(config.n_steps)])
            #NOTE: Assume M vs M and critic_mod_both == True
                if config.crit_both:
                    team_all_MC_targets = []
                    opp_all_MC_targets = []
                    MC_targets_team = np.zeros(env.num_TA)
                    MC_targets_opp = np.zeros(env.num_OA)
                    for n in range(et_i+1):
                        MC_targets_team = team_n_step_rewards[et_i-n] + MC_targets_team*config.gamma
                        team_all_MC_targets.append(MC_targets_team)
                        MC_targets_opp = opp_n_step_rewards[et_i-n] + MC_targets_opp*config.gamma
                        opp_all_MC_targets.append(MC_targets_opp)
                    for n in range(et_i+1):
                        n_step_targets_team = np.zeros(env.num_TA)
                        n_step_targets_opp = np.zeros(env.num_OA)
                        if (et_i + 1) - n >= config.n_steps: # sum n-step target (when more than n-steps remaining)
                            n_step_targets_team = np.sum(np.multiply(np.asarray(team_n_step_rewards[n:n+config.n_steps]),(n_step_gammas)),axis=0)
                            n_step_targets_opp = np.sum(np.multiply(np.asarray(opp_n_step_rewards[n:n+config.n_steps]),(n_step_gammas)),axis=0)

                            n_step_next_ob_team = team_n_step_next_obs[n - 1 + config.n_steps]
                            n_step_done_team = team_n_step_dones[n - 1 + config.n_steps]

                            n_step_next_ob_opp = opp_n_step_next_obs[n - 1 + config.n_steps]
                            n_step_done_opp = opp_n_step_dones[n - 1 + config.n_steps]
                        else: # n-step = MC if less than n steps remaining
                            n_step_targets_team = team_all_MC_targets[et_i-n]
                            n_step_next_ob_team = team_n_step_next_obs[-1]
                            n_step_done_team = team_n_step_dones[-1]

                            n_step_targets_opp = opp_all_MC_targets[et_i-n]
                            n_step_next_ob_opp = opp_n_step_next_obs[-1]
                            n_step_done_opp = opp_n_step_dones[-1]
                        if config.d4pg:
                            default_prio = 5.0
                        else:
                            default_prio = 3.0
                        priorities = np.array([np.zeros(config.k_ensembles) for i in range(env.num_TA)])
                        priorities[:,current_ensembles] = 5.0
                        if config.sil:
                            SIL_priorities = np.ones(env.num_TA)*default_prio
                        

                        exp_team = np.column_stack((np.transpose(team_n_step_obs[n]),
                                            team_n_step_acs[n],
                                            np.expand_dims(team_n_step_rewards[n], 1),
                                            np.expand_dims([n_step_done_team for i in range(env.num_TA)], 1),
                                            np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_team, 1),
                                            np.expand_dims([team_n_step_ws[n] for i in range(env.num_TA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(env.num_TA)],1)))


                        exp_opp = np.column_stack((np.transpose(opp_n_step_obs[n]),
                                            opp_n_step_acs[n],
                                            np.expand_dims(opp_n_step_rewards[n], 1),
                                            np.expand_dims([n_step_done_opp for i in range(env.num_OA)], 1),
                                            np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                            np.expand_dims(n_step_targets_opp, 1),
                                            np.expand_dims([opp_n_step_ws[n] for i in range(env.num_OA)], 1),
                                            priorities,
                                            np.expand_dims([default_prio for i in range(env.num_TA)],1)))
                
                        exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)

                        if exps is None:
                            exps = torch.from_numpy(exp_comb)
                        else:
                            exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)
                    
                    prox_team_tensor = misc.convertProxListToTensor(sortedByProxTeamList, env.num_TA, prox_item_size)
                    prox_opp_tensor = misc.convertProxListToTensor(sortedByProxOppList, env.num_OA, prox_item_size)
                    comb_prox_tensor = torch.cat((prox_team_tensor, prox_opp_tensor), dim=1)
                    # Fill in values for zeros for the hidden state
                    exps = torch.cat((exps[:, :, :], torch.zeros((len(exps), env.num_TA*2, config.hidden_dim_lstm*4), dtype=exps.dtype), comb_prox_tensor.double()), dim=2)
                    #maddpg.get_recurrent_states(exps, obs_dim_TA, acs_dim, env.num_TA*2, hidden_dim_lstm,maddpg.torch_device)
                    shared_exps[int(ep_num[env_num].item())][:len(exps)] = exps
                    exp_i[int(ep_num[env_num].item())] += et_i
                    ep_num[env_num] += 1
                    del exps
                    exps = None
                    torch.cuda.empty_cache()

                #############################################################################################################################################################
                # push exp to queue
                # log
                if ep_i > 1:
                    team_avg_rew = [np.asarray(team_n_step_rewards)[:,i].sum()/float(et_i) for i in range(env.num_TA)] # divide by time step?
                    team_cum_rew = [np.asarray(team_n_step_rewards)[:,i].sum() for i in range(env.num_TA)]
                    opp_avg_rew = [np.asarray(opp_n_step_rewards)[:,i].sum()/float(et_i) for i in range(env.num_TA)]
                    opp_cum_rew = [np.asarray(opp_n_step_rewards)[:,i].sum() for i in range(env.num_TA)]


                    team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                        'why': env.team_envs[0].statusToString(world_stat),
                                                        'agents_kickable_percentages': [(tkc/time_step)*100 for tkc in team_kickable_counter],
                                                        'possession_percentages': [(tpc/time_step)*100 for tpc in team_possession_counter],
                                                        'average_reward': team_avg_rew,
                                                        'cumulative_reward': team_cum_rew},
                                                        ignore_index=True)

                    
                    opp_step_logger_df = opp_step_logger_df.append({'time_steps': time_step, 
                                                        'why': env.opp_team_envs[0].statusToString(world_stat),
                                                        'agents_kickable_percentages': [(okc/time_step)*100 for okc in opp_kickable_counter],
                                                        'possession_percentages': [(opc/time_step)*100 for opc in opp_possession_counter],
                                                        'average_reward': opp_avg_rew,
                                                        'cumulative_reward': opp_cum_rew},
                                                        ignore_index=True)


                # Launch evaluation session
                if ep_i > 1 and ep_i % config.eval_after == 0 and config.eval:
                    thread.start_new_thread(launch_eval,(
                        [load_path + ("agent_%i/model_episode_%i.pth" % (i,ep_i)) for i in range(env.num_TA)], # models directory -> agent -> most current episode
                        eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation_ep" + str(ep_i),
                        7000,env.num_TA,env.num_OA,config.ep_length,config.device,config.use_viewer,))
                if halt.all(): # load when other process is loading buffer to make sure its not saving at the same time
                    ready[env_num] = 1
                    current_ensembles = maddpg.load_random_ensemble(side='team',nagents=env.num_TA,models_path = config.ensemble_path,load_same_agent=config.load_same_agent) # use for per ensemble update counter

                    if config.agent2d and config.preloaded_agent2d:
                        maddpg.load_agent2d(side='opp',load_same_agent=config.load_same_agent,models_path=preload_agent2d_path,nagents=env.num_OA)
                    elif config.agent2d:
                        maddpg.load_agent2d(side='opp',models_path =config.session_path +"models/",load_same_agent=config.load_same_agent,nagents=env.num_OA)  
                    else:
                            
                        if np.random.uniform(0,1) > config.self_play_prob: # self_play_proba % chance loading self else load an old ensemble for opponent
                            maddpg.load_random(side='opp',nagents =env.num_OA,models_path =config.load_path,load_same_agent=config.load_same_agent)
                            pass
                        else:
                            maddpg.load_random_ensemble(side='opp',nagents = env.num_OA,models_path = config.ensemble_path,load_same_agent=config.load_same_agent)
                            pass

                    if config.left_agent2d:
                        maddpg.load_agent2d(side='team',load_same_agent=config.load_same_agent,models_path=preload_agent2d_path,nagents=env.num_OA)

                    while halt.all():
                        time.sleep(0.1)
                    total_dim = (obs_dim + env.acs_dim + 5) + config.k_ensembles + 1 + (config.hidden_dim_lstm*4) + prox_item_size
                    ep_num.copy_(torch.zeros_like(ep_num,requires_grad=False))
                    [s.copy_(torch.zeros(config.max_num_exps,2*env.num_TA,total_dim)) for s in shared_exps[:int(ep_num[env_num].item())]] # done loading
                    del exps
                    exps = None

                end = time.time()
                print(end-start)

                break
            elif d:
                break            
                
            team_obs = team_next_obs
            opp_obs = opp_next_obs


    
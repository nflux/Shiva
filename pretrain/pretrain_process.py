import os, sys
import numpy as np
import utils.misc as misc
import time
import torch
from pathlib import Path
from utils.buffers import ReplayBufferLSTM
import utils.buffers as buff
import algorithms.maddpg as mad_algo
import torch.multiprocessing as mp
from multiprocessing import Pool
import gc
import algorithms.updates as updates

class pretrain:

    def __init__(self, config, env):
        self.env = env
        self.config = config
        self.numBuffs = config.num_buffs
        self.num_TA = config.num_left
        self.acs_dim = config.ac_dim
        self.obs_dim = env.obs_dim
        self.prox_item_size = env.prox_item_size
        self.pt = config.pt
        self.use_pt_data = config.use_pt_data
        self.episode_length = config.ep_length
        self.n_steps = config.n_steps
        self.gamma = config.gamma       
        self.to_gpu = config.to_gpu  
        self.D4PG = config.d4pg
        self.SIL = config.sil
        self.k_ensembles = config.k_ensembles
        self.push_only_left = config.push_only_left
        self.LSTM_policy = config.lstm_pol
        self.pt_update_cycles = config.pt_cycles
        self.use_cuda = config.cuda
        self.batch_size = config.batch_size
        self.number_of_updates = config.num_updates
        self.load_path = config.load_path
        self.ensemble_path = config.ensemble_path
        self.forward_pass = config.forward_pass
        self.seq_length = config.seq_length
        self.SIL_update_ratio = config.update_ratio
        self.load_same_agent = config.load_same_agent
        self.multi_gpu = config.multi_gpu
        self.session_path = config.session_path
        self.data_parallel = config.data_parallel
        self.lstm_burn_in = config.burn_in_lstm
        self.LSTM = config.lstm_crit
        self.all_l = []
        self.all_r = []
        self.all_s = []
        self.pt_threads = []

        #Begin the pretrain process
        self.pretraining()
            


    def pretraining(self):  
        if self.pt or self.use_pt_data:
            # ------------------------------------ Start Pretrain --------------------------------------------
            pt_trbs = [ReplayBufferLSTM(self.config,self.obs_dim,self.prox_item_size,self.pt) for _ in range(self.numBuffs)]
            pt_orbs = [ReplayBufferLSTM(self.config,self.obs_dim,self.prox_item_size,self.pt) for _ in range(self.numBuffs)]
            
            # ------------ Load in shit from csv into buffer ----------------------
            print("Load PT Buffers")
            self.load_PT_Buffers()

        if self.pt:

            # ------------ Done loading shit --------------------------------------
            # ------------ Pretrain actor/critic same time---------------
            #define/update the noise used for exploration

            mad_algo.scale_noise(0.0)
            mad_algo.reset_noise()
            mad_algo.scale_beta(1)
            update_session = 999999

            self.update_cycles()


    def load_PT_Buffers(self) :
            for i in range(1,self.numBuffs+1):
                left_files = []
                right_files = []
                
                if os.path.isdir(os.getcwd() + '/pretrain_data/pt_logs_%i' % ((i+1)*1000)):
                    team_files = os.listdir(os.getcwd() + '/pretrain_data/pt_logs_%i' % ((i+1)*1000))
                    left_files = [os.getcwd() + '/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + f for f in team_files if '_left_' in f]
                    right_files = [os.getcwd() + '/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + f for f in team_files if '_right_' in f]
                    status_file = os.getcwd() + '/pretrain_data/pt_logs_%i/' % ((i+1)*1000) + 'log_status.csv'
                else:
                    print('log directory DNE')
                    exit(0)
                self.all_l.append(left_files)
                self.all_r.append(right_files)
                self.all_s.append(status_file)

            second_args = [(self.num_TA,self.obs_dim,self.acs_dim,pt_trbs[i],pt_orbs[i],self.episode_length,self.n_steps,self.gamma,self.D4PG,self.SIL,self.k_ensembles,self.push_only_left,self.LSTM_policy,self.prox_item_size) for i in range(self.num_buffs)]
            
            with Pool() as pool:
                pt_rbs = pool.starmap(misc.load_buffer, zip(self.all_l,self.all_r,self.all_s, second_args))
            pt_trbs,pt_orbs = list(zip(*pt_rbs))
            
            del pt_rbs
            
            for i in range(self.numBuffs):
                print("Length of RB_",i,": ",len(pt_trbs[i]))
                
            #team_replay_buffer = pt_trbs[0]
            #opp_replay_buffer = pt_orbs[0]
            
            pt_trb = ReplayBufferLSTM(self.config,self.obs_dim,self.prox_item_size,self.pretrain)
            pt_orb = ReplayBufferLSTM(self.config,self.obs_dim,self.prox_item_size,self.pretrain)
            for j in range(self.numBuffs):
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


    def update_cycles():
        for u in range(self.pt_update_cycles):
                print("PT Update Cycle: ",u)
                print("PT Completion: ",(u*100.0)/(float(self.pt_update_cycles)),"%")
                rb_i = np.random.randint(self.numBuffs)


                threads = []
                for a_i in range(1):
                    threads.append(mp.Process(target=updates.imitation_thread,args=(a_i,self.to_gpu,len(pt_trb),self.batch_size,
                        pt_trb,pt_orb,number_of_updates,
                        self.load_path,self.update_session,self.ensemble_path,self.forward_pass,self.LSTM,self.LSTM_policy,self.seq_length,self.k_ensembles,self.SIL,self.SIL_update_ratio,self.num_TA,self.load_same_agent,self.multi_gpu,self.session_path,self.data_parallel,self.lstm_burn_in)))
                [thr.start() for thr in threads]
            
                agentID = 0
                buffer_size = len(pt_trb)

                number_of_updates = 5000
                batches_to_sample = 50
                if len(pt_trb) < batch_size*(batches_to_sample):
                    batches_to_sample = 1
                priorities = [] 
                start = time.time()
                for ensemble in range(self.k_ensembles):

                    mad_algo.load_same_ensembles(self.ensemble_path,self.ensemble,mad_algo.nagents_team,load_same_agent=self.load_same_agent)

                    #start = time.time()
                    for up in range(number_of_updates):
                        offset = up % batches_to_sample

                        if not self.load_same_agent:
                            inds = pt_trb.get_PER_inds(agentID,self.batch_size,self.ensemble)
                        else:
                            if up % batches_to_sample == 0:
                                m = 0
                                #inds = pt_trb[rb_i].get_PER_inds(m,batches_to_sample*batch_size,ensemble)
                                inds = np.random.choice(np.arange(len(pt_trb)), size=self.batch_size*batches_to_sample, replace=False)

                                #if len(priorities) > 1:
                                    #for c,p in enumerate(priorities):
                                    #    pt_trb.update_priorities(agentID=m,inds = inds[c*batch_size:(c+1)*batch_size], prio=p,k = ensemble) 
                                priorities = [] 


                        # FOR THE LOVE OF GOD DONT USE TORCH TO GET INDICES

                        if LSTM:
                            team_sample = pt_trb.sample(inds[self.batch_size*offset:self.batch_size*(offset+1)],to_gpu=self.to_gpu,device=mad_algo.torch_device)
                            opp_sample = pt_orb.sample(inds[self.batch_size*offset:self.batch_size*(offset+1)],to_gpu=self.to_gpu,device=mad_algo.torch_device)
                            priorities=mad_algo.pretrain_critic_LSTM(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',load_same_agent=self.load_same_agent,lstm_burn_in=self.lstm_burn_in)
                            #print("Use pretrain function")
                            #pt_trb.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)

                            if not load_same_agent:
                                print("No implementation")
                        else:
                            team_sample = pt_trb.sample(inds[self.batch_size*offset:self.batch_size*(offset+1)],
                                                        to_gpu=self.to_gpu,norm_rews=False,device=mad_algo.torch_device)
                            opp_sample = pt_orb.sample(inds[self.batch_size*offset:self.batch_size*(offset+1)],
                                                        to_gpu=self.to_gpu,norm_rews=False,device=mad_algo.torch_device)
                            if not load_same_agent:
                                print("No implementation")
                            else:
                            
                                priorities.append(mad_algo.pretrain_critic_MC(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=self.forward_pass,load_same_agent=self.load_same_agent,session_path=self.session_path))
                                #priorities.append(mad_algo.pretrain_critic(team_sample=team_sample, opp_sample=opp_sample, agent_i =agentID, side='team',forward_pass=forward_pass,load_same_agent=load_same_agent,session_path=session_path))
                                #team_replay_buffer.update_priorities(agentID=m,inds = inds, prio=priorities,k = ensemble)
                                if up % number_of_updates/10 == 0: # update target half way through
                                    mad_algo.update_agent_targets(0,number_of_updates/10)
                        if SIL:
                            for i in range(self.SIL_update_ratio):
                                inds = pt_trb.get_SIL_inds(agentID=m,batch_size=self.batch_size)
                                team_sample = pt_trb.sample(inds,
                                                        to_gpu=self.to_gpu,norm_rews=False)
                                opp_sample = pt_orb.sample(inds,
                                                        to_gpu=self.to_gpu,norm_rews=False)
                                priorities = mad_algo.SIL_update(team_sample, opp_sample, agentID, 'team') # 
                                pt_trb.update_SIL_priorities(agentID=m,inds = inds, prio=priorities)
                    #for c,p in enumerate(priorities):
                    #    if c != (len(priorities)-1):
                    #        pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):(batch_size*(c+1))+(-batch_size*batches_to_sample)], prio=p,k = ensemble) 
                    #    else:
                    #        pt_trb.update_priorities(agentID=m,inds = inds[(-batch_size*batches_to_sample)+(batch_size*c):], prio=p,k = ensemble) 
                    #print(time.time()-start)
                    if not load_same_agent:
                        mad_algo.update_agent_targets(agentID,number_of_updates)
                        mad_algo.save_agent(self.load_path,self.update_session,agentID)
                        mad_algo.save_ensemble(ensemble_path,ensemble,agentID)
                    else:
                        mad_algo.update_agent_hard_critic(0)
                        print(time.time()-start,"<-- Critic Update Cycle")

                        [thr.join() for thr in threads]
                        mad_algo.load_ensemble_policy(self.ensemble_path,self.ensemble,0) # load only policy from updated policy thread
                        mad_algo.update_agent_hard_policy(agentID=0)
                        [mad_algo.save_agent(self.load_path,self.update_session,i,self.load_same_agent,mad_algo.torch_device) for i in range(self.num_TA)]
                        [mad_algo.save_ensemble(self.ensemble_path,self.ensemble,i,self.load_same_agent,mad_algo.torch_device) for i in range(self.num_TA)]
                    print(time.time()-start,"<-- Full Cycle")

                mad_algo.update_hard_policy() 
                mad_algo.update_hard_critic()
                mad_algo.save_agent2d(load_path,0,self.load_same_agent,mad_algo.torch_device)

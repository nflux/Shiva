

class RoboEnvs(rc_env):
    def __init__(self, config, args):
        self.config = config
        self.env = super().__init__(config)
        self.obs_dim = self.env.team_num_features
        self.maddpg = MADDPG.init(config, self.env)

        self.prox_item_size = num_TA*(2*obs_dim_TA + 2*acs_dim)
        self.team_replay_buffer = ReplayBuffer(replay_memory_size , num_TA,
                                            obs_dim_TA,acs_dim,batch_size, LSTM, seq_length,overlap,hidden_dim_lstm,k_ensembles, prox_item_size, SIL)

        self.opp_replay_buffer = ReplayBuffer(replay_memory_size , num_TA,
                                            obs_dim_TA,acs_dim,batch_size, LSTM, seq_length,overlap,hidden_dim_lstm,k_ensembles, prox_item_size, SIL)
        self.max_episodes_shared = 30
        self.total_dim = (obs_dim_TA + acs_dim + 5) + k_ensembles + 1 + (hidden_dim_lstm*4) + prox_item_size

        self.shared_exps = [[torch.zeros(max_num_experiences,2*num_TA,total_dim,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
        self.exp_indices = [[torch.tensor(0,requires_grad=False).share_memory_() for _ in range(max_episodes_shared)] for _ in range(num_envs)]
        self.ep_num = torch.zeros(num_envs,requires_grad=False).share_memory_()

        self.halt = Variable(torch.tensor(0).byte()).share_memory_()
        self.ready = torch.zeros(num_envs,requires_grad=False).byte().share_memory_()
        self.update_counter = torch.zeros(num_envs,requires_grad=False).share_memory_()
    
    def run(self):
        processes = []
        for i in range(num_envs):
            processes.append(mp.Process(target=run_envs, args=(seed + (i * 100), port + (i * 1000), shared_exps[i],
                                        exp_indices[i],i,ready,halt,update_counter,(history+str(i)),ep_num)))
        
        for p in processes: # Starts environments
            p.start()
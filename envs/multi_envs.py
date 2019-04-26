import utils.buffers as buff
from rc_env import rc_env
import torch
from torch.autograd import Variable

class RoboEnvs(rc_env):
    def __init__(self, config):
        self.config = config
        self.env = super().__init__(config)
        self.obs_dim = self.env.team_num_features
        # self.maddpg = MADDPG.init(config, self.env)

        self.prox_item_size = num_TA*(2*self.obs_dim + 2*config.ac_dim)
        self.team_replay_buffer = buff.init_buffer(config, config.lstm_crit or config.lstm_pol,
                                                    self.obs_dim, self.prox_item_size)

        self.opp_replay_buffer = buff.init_buffer(config, config.lstm_crit or config.lstm_pol,
                                                    self.obs_dim, self.prox_item_size)
        self.max_episodes_shared = 30
        self.total_dim = (self.obs_dim + config.ac_dim + 5) + config.k_ensembles + 1 + (config.hidden_dim_lstm*4) + self.prox_item_size

        self.shared_exps = [[torch.zeros(max_num_experiences,2*num_TA,total_dim,requires_grad=False).share_memory_() for _ in range(self.max_episodes_shared)] for _ in range(config.num_envs)]
        self.exp_indices = [[torch.tensor(0,requires_grad=False).share_memory_() for _ in range(self.max_episodes_shared)] for _ in range(config.num_envs)]
        self.ep_num = torch.zeros(config.num_envs,requires_grad=False).share_memory_()

        self.halt = Variable(torch.tensor(0).byte()).share_memory_()
        self.ready = torch.zeros(config.num_envs,requires_grad=False).byte().share_memory_()
        self.update_counter = torch.zeros(config.num_envs,requires_grad=False).share_memory_()
    
    def run(self):
        processes = []
        for i in range(self.config.num_envs):
            processes.append(mp.Process(target=self.env.run_envs, args=(self.env.seed + (i * 100), self.env.port + (i * 1000), self.shared_exps[i],
                                        self.exp_indices[i],i,self.ready,self.halt,self.update_counter,(config.history+str(i)),self.ep_num)))
        
        for p in processes: # Starts environments
            p.start()
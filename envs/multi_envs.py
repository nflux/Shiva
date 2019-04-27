import envs.rc_env as rc
import utils.buffers as buff
import torch
from torch.autograd import Variable
import torch.multiprocessing as mp

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

        for i in range(self.config.num_envs):
            processes.append(mp.Process(target=envs[i].run_env, args=(self.shared_exps[i],
                                        self.exp_indices[i],i,self.ready,self.halt,self.update_counter,(self.config.history+str(i)),self.ep_num)))

        for p in processes: # Starts environments
            p.start()
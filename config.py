import ast
import torch
import utils.misc as misc
import datetime

def get_dict(config):
    conf_dict = {}

    sections = config.sections()

    for section in sections:
        options = config.options(section)
        for option in options:
            conf_dict[option] = ast.literal_eval(config.get(section, option))
    
    return conf_dict

class Config:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)

class RoboConfig(Config):
    def __init__(self, config):
        self.conf_dict = get_dict(config)
        self.conditional_inits()

        num_TA = self.conf_dict['num_left']
        num_OA = self.conf_dict['num_right']
        # Prep Session Files ------------------------------
        self.session_path = None
        current_day_time = datetime.datetime.now()
        self.session_path = 'data/training_sessions/' + \
                                        str(current_day_time.month) + \
                                        '_' + str(current_day_time.day) + \
                                        '_' + str(current_day_time.hour) + '_' + \
                                        str(num_TA) + '_vs_' + str(num_OA) + "/"
        self.hist_dir = self.session_path +"history"
        self.eval_hist_dir = self.session_path +"eval_history"
        self.eval_log_dir = self.session_path +"eval_log" # evaluation logfiles
        self.load_path = self.session_path +"models/"
        self.ensemble_path = self.session_path +"ensemble_models/"
        misc.prep_session(self.session_path,self.hist_dir,self.eval_hist_dir,self.eval_log_dir,
                            self.load_path,self.ensemble_path,self.conf_dict['log'],num_TA)

        super().__init__(self.conf_dict)

    def conditional_inits(self):
        if self.conf_dict['d4pg']:
            self.conf_dict['a_lr'] = 0.0001 # actor learning rate
            self.conf_dict['c_lr'] = 0.001 # critic learning rate
        else:
            self.conf_dict['a_lr'] = 0.0002
            self.conf_dict['c_lr'] = 0.0001
        self.conf_dict['delta_z'] = (float(self.conf_dict['vmax'] - self.conf_dict['vmin'])) / (self.conf_dict['n_atoms'] - 1)
        self.conf_dict['freeze_actor'] = 0.0
        self.conf_dict['freeze_critic'] = 0.0

        if self.conf_dict['load_same_agent']:
            self.conf_dict['num_update_threads'] = 1
        
        self.conf_dict['overlap'] = self.conf_dict['seq_length'] // 2

        if self.conf_dict['seq_length'] % 2 != 0:
            print('Sequnce length must be divisble by 2')
            exit(0)
        
        if self.conf_dict['cuda']:
            self.conf_dict['device'] = 'cuda'
            self.conf_dict['to_gpu'] = True
        else:
            self.conf_dict['device'] = 'cpu'
            self.conf_dict['to_gpu'] = False
            torch.set_num_threads(self.conf_dict['num_threads'])
        self.conf_dict['torch_device'] = torch.device(self.conf_dict['device'])
        
        if self.conf_dict['al'] == 'high':
            self.conf_dict['discrete_action'] = True
        else:
            self.conf_dict['discrete_action'] = False
        
        self.conf_dict['initial_models'] = ["data/training_sessions/5_21_15_3_vs_3/ensemble_models/ensemble_agent_0/model_0.pth",
                                            "data/training_sessions/5_21_15_3_vs_3/ensemble_models/ensemble_agent_1/model_0.pth",
                                            "data/training_sessions/5_21_15_3_vs_3/ensemble_models/ensemble_agent_2/model_0.pth"]

        self.conf_dict['burn_in_eps'] = float(self.conf_dict['burn_in']) / self.conf_dict['untouched']

        self.conf_dict['current_ensembles'] = [0]*self.conf_dict['num_left']
    
    def env_inits(self, env):
        team_net_params = []
        for acsp, obsp in zip([env.action_list for i in range(env.num_TA)], env.team_obs):
            if self.preprocess:
                num_in_pol = config.reduced_obs_dim
            else:
                num_in_pol = obsp.shape[0]
            num_in_reducer = obsp.shape[0]
            num_out_pol =  len(env.action_list)

            if not self.discrete_action:
                num_out_pol = len(env.action_list) + len(env.team_action_params[0])
            
            num_in_EM = (num_out_pol*env.num_TA) + num_in_pol
            num_out_EM = num_in_pol

            num_in_critic = (num_in_pol - num_out_pol)  + (num_out_pol * env.num_TA *2 ) + (env.num_TA -1)            
            
            team_net_params.append({'num_in_pol': num_in_pol,
                                    'num_out_pol': num_out_pol,
                                    'num_in_critic': num_in_critic,
                                    'num_in_EM': num_in_EM,
                                    'num_out_EM': num_out_EM,
                                    'num_in_reducer': num_in_reducer})

        setattr(self, 'team_net_params', team_net_params)
    
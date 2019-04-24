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
        session_path = None
        current_day_time = datetime.datetime.now()
        session_path = 'training_sessions/' + \
                                        str(current_day_time.month) + \
                                        '_' + str(current_day_time.day) + \
                                        '_' + str(current_day_time.hour) + '_' + \
                                        str(num_TA) + '_vs_' + str(num_OA) + "/"
        hist_dir = session_path +"history"
        eval_hist_dir = session_path +"eval_history"
        eval_log_dir = session_path +"eval_log" # evaluation logfiles
        load_path = session_path +"models/"
        ensemble_path = session_path +"ensemble_models/"
        misc.prep_session(session_path,hist_dir,eval_hist_dir,eval_log_dir,
                            load_path,ensemble_path,self.conf_dict['log'],num_TA)

        super().__init__(self.conf_dict)

    def conditional_inits(self):
        if self.conf_dict['d4pg']:
            self.conf_dict['a_lr'] = 0.0001 # actor learning rate
            self.conf_dict['c_lr'] = 0.001 # critic learning rate
        else:
            self.conf_dict['freeze_actor'] = 0.0
            self.conf_dict['freeze_critic'] = 0.0
            self.conf_dict['a_lr'] = 0.0002
            self.conf_dict['c_lr'] = 0.0001
        
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
        
        if self.conf_dict['al'] == 'high':
            self.conf_dict['discrete_action'] = True
        else:
            self.conf_dict['discrete_action'] = False
    
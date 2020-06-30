import time
import torch
import torch.nn
import random
from shiva.core.admin import logger

class Agent:

    id = None
    step_count = 0
    done_count = 0
    num_updates = 0
    num_evolutions = {'truncate': 0, 'perturb': 0, 'resample': 0}
    role = None
    
    def __init__(self, id, obs_space, acs_space, configs):
        '''
        Base Attributes of Agent
            agent_id = given by the learner
            observation_space
            act_dim
            policy = Neural Network Policy
            target_policy = Target Neural Network Policy
            optimizer = Optimizer Function
            learning_rate = Learning Rate
        '''
        self.configs = configs
        {setattr(self, k, v) for k, v in self.configs['Agent'].items()}
        self.id = id
        self.agent_config = self.configs['Agent']
        self.networks_config = self.configs['Network']
        self.step_count = 0
        self.done_count = 0
        self.num_updates = 0
        self.role = self.agent_config['role'] if 'role' in self.agent_config else 'Unassigned Role' # use 'A' for the folder name when there's no role assigned
        self.obs_space = obs_space
        self.acs_space = acs_space
        try:
            self.optimizer_function = getattr(torch.optim, self.agent_config['optimizer_function'])
        except:
            self.log("No optimizer", to_print=True)
        self.policy = None
        self.device = torch.device('cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_attrs = ['step_count', 'done_count', 'num_updates', 'num_evolutions', 'role']
        self.net_names = []
        self.hps = []
        self.save_filename = "{id}.state"

    def reset_noise(self):
        pass

    def __str__(self):
        return "<{}:id={}>".format(self.__class__, self.id)

    def instantiate_networks(self):
        raise NotImplemented

    def to_device(self, device):
        assert hasattr(self, 'net_names'), "Need this attribute to turn networks to a device"
        self.device = device
        for net_name in self.net_names:
            if 'optimizer' in net_name:
                continue
            nn = getattr(self, net_name)
            nn.to(self.device)

    def save(self, save_path, step):
        '''
            During saving maintain the .pth file name to have the same name as the Agent attribute
                torch.save(self.policy, save_path + '/policy.pth')
                torch.save(self.critic, save_path + '/critic.pth')
        '''
        raise NotImplemented

    def load_net(self, policy_name, policy_file):
        setattr(self, policy_name, torch.load(policy_file))

    def save_state_dict(self, save_path):
        assert hasattr(self, 'net_names'), "Need this attribute to save"
        dict = {}
        # save general state attributes
        for attr in self.state_attrs:
            dict[attr] = getattr(self, attr)
        # save networks
        for net_name in self.net_names:
            net = getattr(self, net_name)
            dict[net_name] = net.state_dict()
        # save hyperparameters
        for hp in self.hps:
            dict[hp] = getattr(self, hp)
        dict['class_module'], dict['class_name'] = self.get_module_and_classname()
        dict['inits'] = (self.id, self.obs_space, self.acs_space, self.configs)
        filename = save_path + '/' + self.save_filename.format(id=self.id)
        torch.save(dict, filename)

    def load_state_dict(self, state_dict):
        '''Assuming @agent has all the attributes already and @state_dict contains expected keys for that @agent'''
        # load general state attributes
        for attr in self.state_attrs:
            setattr(self, attr, state_dict[attr])
        # load networks
        for net_name in self.net_names:
            net = getattr(self, net_name)
            net.load_state_dict(state_dict[net_name])
        # load hyperparameters
        for hp in self.hps:
            setattr(self, hp, state_dict[hp])

    def get_action(self, obs):
        raise NotImplemented

    def get_metrics(self):
        raise NotImplemented

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['Agent']:
            text = '{}\t{}'.format(self, msg)
            logger.info(text, to_print or self.configs['Admin']['print_debug'])

    @staticmethod
    def copy_model_over(from_model, to_model):
        """
            Copies model parameters from @from_model to @to_model
        """
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    @staticmethod
    def mod_lr(optim, lr):
        for g in optim.param_groups:
            # print(g['lr'])
            g['lr'] = lr

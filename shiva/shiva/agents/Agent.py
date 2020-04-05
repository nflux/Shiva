import time
import torch
import torch.nn
import random
from shiva.core.admin import logger

class Agent:
    def __init__(self, id, obs_space, acs_space, agent_config, networks_config):
        super(Agent, self).__init__()
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
        {setattr(self, k, v) for k,v in agent_config.items()}
        self.id = id
        self.agent_config = agent_config
        self.networks_config = networks_config
        self.step_count = 0
        self.done_count = 0
        self.num_updates = 0
        self.role = agent_config['role'] if 'role' in agent_config else 'Role' # use 'A' for the folder name when there's no role assigned
        self.obs_space = obs_space
        self.acs_space = acs_space
        try:
            self.optimizer_function = getattr(torch.optim, agent_config['optimizer_function'])
        except:
            self.log("No optimizer", to_print=True)
        self.policy = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_attrs = ['step_count', 'done_count', 'num_updates', 'role']
        self.save_filename = "{id}.state"

    def reset_noise(self):
        pass

    def __str__(self):
        return "<{}:id={}>".format(self.__class__, self.id)

    def instantiate_networks(self):
        raise NotImplemented

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
        # save networks
        for net_name in self.net_names:
            net = getattr(self, net_name)
            dict[net_name] = net.state_dict()
        # save other state attributes
        for attr in self.state_attrs:
            dict[attr] = getattr(self, attr)
        dict['class_module'], dict['class_name'] = self.get_module_and_classname()
        dict['inits'] = (self.id, self.obs_space, self.acs_space, self.agent_config, self.networks_config)
        filename = save_path + self.save_filename.format(id=self.id)
        torch.save(dict, filename)

    def get_action(self, obs):
        raise NotImplemented

    def get_metrics(self):
        raise NotImplemented

    def log(self, msg, to_print=False):
        text = '{}\t{}'.format(self, msg)
        logger.info(text, to_print)

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

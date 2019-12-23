import Environment
import GymEnvironment
import torch
import torch.multiprocessing as mp

class MultiGymWrapper(Environment):
    def __init__(self,configs,queue,agent,episode_count):
        super(MultiGymWrapper,configs).__init__(configs)
        self.queue = queue
        self.master_agent = agent
        self.agent = copy.deepcopy(self.master_agent)
        self.episode_count = 0
        self.process_list = list()
        self.launch_envs()
        self.launch_processes()
        self.launch_wrapper_process()



    def step(actions):

        pass

    def launch_envs(self):
        environment = getattr(envs, self.configs['Environment']['type'])
        self.envs = [environment(self.configs['Environment'])] * self.instances
        self.shared_tensor = torch.zeros(self.instances, max(self.envs[0].observation_space,self.envs[0].action_space)).shared_memory()
        self.step_done = torch.zeros(self.instances).shared_memory()
        self.episode_done = torch.zeros(self.instances).shared_memory()
        self.done_count = 0


    def launch_processes(self):
        for env in self.envs:
            p = mp.Process(target = process_target, args = (env,self.shared_tensor,self.step_done,self.episode_done))
            p.start()
            self.process_list.append(p) 


    def process_target(self,env,shared_tensor,step_done,episode_done):
        epsiode_trajectory = torch.zeros(self.max_episode_length)


        pass

    def wrapper_target(self,shared_tensor,step_done,episode_done):


        pass

from .Environment import Environment
from .GymEnvironment import GymEnvironment
import envs
import torch
import torch.multiprocessing as mp
import copy

class MultiGymWrapper(Environment):
    def __init__(self,configs,queue,agent,episode_count):
        super(MultiGymWrapper,self).__init__(configs)
        self.queue = queue
        self.master_agent = agent
        self.agent = copy.deepcopy(self.master_agent)
        self.process_list = list()
        self.episode_count = episode_count
        self.step_count = 0
        self.configs = configs




    def step(self):
        while(self.stop_collecting.item() == 0):
            if
            if(self.step_control.sum().item() == self.num_instances):
                actions = torch.tensor([self.agent.get_action(obs) for obs in self.observations.numpy()])
                self.observations[:,0:self.envs[0].action_space] = actions
                self.step_control.fill_(0)







    def process_target(self,env,observations,step_control,stop_collecting,id,queue):
        #tensor for storing episodes per process/env
        episode_trajectory = torch.zeros(self.max_episode_length)
        episode_index= 0
        env.reset()
        observation = env.get_observation()
        observations[id] = torch.tensor(observation).float()
        step_control[id] = 1
        while(stop_collecting.item() == 0):
            if(step_control[id] == 0):
                action = observations[id][:env.action_space]
                next_observation, reward, done, more_data = env.step(action)
                t = [observation, action, reward, next_observation, int(done)]
                exp = copy.deepcopy(t)
                episode_trajectory[episode_index] = exp
                episode_index +=1
                if done:
                    env.reset()
                    observation = env.get_observation()
                    observations[id] = observation
                    step_control[id] = 1
                else:
                    observations[id] = next_observation
                    observation = next_observation
                    step_control[id] = 1


    def launch_processes(self):
        for i in range(self.num_instances):
            print(i)
            p = mp.Process(target = self.process_target, args = (self.envs[i],self.observations,self.step_control,self.stop_collecting,i,self.queue))
            p.start()
            self.process_list.append(p)


    def launch_envs(self):
        environment = getattr(envs, self.configs['sub_type'])
        #list of the environments to be used for episode collection
        self.envs = [environment(self.configs)] * self.num_instances
        #Shared tensor will be used for communication between environment wrapper process and individual environment processes
        self.observations = torch.zeros(self.num_instances, max(self.envs[0].observation_space,self.envs[0].action_space)).share_memory_()
        #Shared tensor will let control data flow through the tensor
        # 0 signals env to process, 1 signals multi wrapper to process
        self.step_control = torch.zeros(self.num_instances).share_memory_()
        #Shared tensor will signal to the envs when to stop collecting episodes
        self.stop_collecting = torch.zeros(1).share_memory_()
        self.done_count = 0
        self.launch_processes()
        self.step()

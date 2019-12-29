from .Environment import Environment
from .GymEnvironment import GymEnvironment
import numpy as np
import envs
import torch
import torch.multiprocessing as mp
import copy

class MultiGymWrapper(Environment):
    def __init__(self,configs,queue,agent,episode_count,agent_dir,total_episodes):
        super(MultiGymWrapper,self).__init__(configs)
        self.queue = queue
        self.master_agent = agent
        self.agent = copy.deepcopy(self.master_agent)
        self.process_list = list()
        self.episode_count = episode_count
        self.total_episodes = total_episodes
        self.step_count = 0
        self.configs = configs
        self.agent_dir = agent_dir




    def step(self):
        while(self.stop_collecting.item() == 0):
            #if self.episode_count % self.agent_update_episodes == 0 and self.episode_count != 0:
                #self.agent.load(self.agent_dir)

            if(self.step_control.sum().item() == self.num_instances):
                observations = self.observations.numpy()
                actions = torch.tensor([self.agent.get_action(obs) for obs in observations])
                self.observations[:,0:self.envs[0].action_space] = actions
                self.step_control.fill_(0)

            if self.episode_count == self.total_episodes:
                self.stop_collecting[0] = 1


        for env in self.envs:
            env.close()

        for p in self.process_list:
            p.join()

        del(self.process_list)


    def process_target(self,env,observations,step_control,stop_collecting,id,queue,max_ep_length):
        #tensor for storing episodes per process/env
        step_count = 1
        ep_observations,ob_idx = np.zeros((max_ep_length,env.observation_space)), 0
        ep_actions, acs_idx = np.zeros((max_ep_length,env.action_space)), 0
        ep_rewards, rew_idx = np.zeros((max_ep_length,1)), 0
        ep_next_observations,nob_idx = np.zeros((max_ep_length,env.observation_space)),0
        ep_dones, done_idx = np.zeros((max_ep_length,1)),0
        env.reset()
        observation = env.get_observation()
        observations[id] = torch.tensor(observation).float()
        step_control[id] = 1
        while(stop_collecting.item() == 0):
            #print('Hello')
            #print('Process: ', step_control[id])
            if(step_control[id] == 0):
                action = observations[id][:env.action_space].numpy()
                next_observation, reward, done, more_data = env.step(action)
                ep_observations[ob_idx:ob_idx+env.observation_space] = observation
                ep_actions[acs_idx:acs_idx+env.action_space] = action
                ep_rewards[rew_idx] = reward
                ep_next_observations[nob_idx:nob_idx+env.observation_space] = next_observation
                ep_dones[done_idx] = int(done)
                ob_idx += env.observation_space
                acs_idx += env.action_space
                rew_idx += 1
                nob_idx += env.observation_space
                done_idx += 1

                '''t = [observation, action, reward, next_observation, int(done)]
                exp = copy.deepcopy(t)
                print('List: ',exp)
                print('Tensor: ', torch.tensor(exp))
                episode_trajectory[episode_index:episode_index + len(exp)] = torch.tensor(exp)
                episode_index += len(exp)'''
                if done:
                    exp = copy.deepcopy(zip(ep_observations[:ob_idx],ep_actions[:acs_idx],ep_rewards[:rew_idx].tolist(),ep_next_observations[:nob_idx],ep_dones[:done_idx].tolist()))
                    queue.put(exp)
                    env.reset()
                    observation = env.get_observation()
                    observations[id] = torch.from_numpy(observation)
                    ep_observations.fill(0)
                    ep_actions.fill(0)
                    ep_rewards.fill(0)
                    ep_next_observations.fill(0)
                    ep_dones.fill(0)
                    ob_idx,acs_idx,rew_idx,nob_idx,done_idx = 0,0,0,0,0
                    step_control[id] = 1
                else:
                    observations[id] = torch.from_numpy(next_observation)
                    observation = next_observation
                    step_control[id] = 1


    def launch_processes(self):
        for i in range(self.num_instances):
            p = mp.Process(target = self.process_target, args = (self.envs[i],self.observations,self.step_control,self.stop_collecting,i,self.queue,self.max_episode_length))
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

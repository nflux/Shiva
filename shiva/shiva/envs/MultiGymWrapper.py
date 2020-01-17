from .Environment import Environment
from shiva.helpers.config_handler import load_class
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions import Categorical
from torch.distributions.normal import Normal
import copy
import time
import os

class MultiGymWrapper(Environment):
    def __init__(self,configs,queue,agent,episode_count,agent_dir,total_episodes, saveLoadFlag, waitForLearner, step_count):
        super(MultiGymWrapper,self).__init__(configs)
        self.queue = queue
        self.agent = agent
        self.process_list = list()
        self.episode_count = episode_count
        self.total_episodes = total_episodes
        self.step_count = step_count
        self.configs = configs
        self.agent_dir = agent_dir
        self.saveLoadFlag = saveLoadFlag
        self.action_available = torch.zeros(1).share_memory_()
        self.waitForLearner = waitForLearner
        self.envs = []
        self.p = mp.Process(target = self.launch_envs)
        self.p.start()




    def step_without_log_probs(self):

        loaded = False
        last_load = 0
        last_ep = 0

        while(self.stop_collecting.item() == 0):

            if self.episode_count.item() > last_ep:
                self.agent.ou_noise.reset()
                last_ep = self.episode_count.item()

            if not loaded:
                # if self.episode_count % self.agent_update_episodes == 0 and self.episode_count != 0:
                if self.episode_count % 5 == 0 and self.episode_count != 0:
                    loaded = True
                    if self.saveLoadFlag.item() == 1:
                        self.agent.load(self.agent_dir)
                        self.agent.actor.eval()
                        self.agent.ou_noise.reset()
                        print("Agent Loaded")
                        self.saveLoadFlag[0] = 0
                    last_load = self.episode_count.item()

            # if self.episode_count % self.agent_update_episodes != 0:
            if self.episode_count > last_load:
                loaded = False

            # if (self.step_control.sum().item() == self.num_instances and self.action_available.item() == 0) or self.step_count.item() == 0:
            if self.step_control.sum().item() == self.num_instances or self.step_count.item() == 0:
                self.actions = self.agent.get_action(copy.deepcopy(self.observations[:,:self.obs_dim]), self.step_count.item())


                '''
                for action in action
                    add noise to each action from the corresponding noise object in the agent?
                
                '''
                # self.action_available[0] = 1

            # if self.action_available.item() == 1:
                self.observations[:,0:self.acs_dim] = copy.deepcopy(self.actions)
                self.control = self.step_control.fill_(0)
                # self.action_available[0] = 0

            if self.episode_count == self.total_episodes:
                self.stop_collecting[0] = 1


        for env in self.envs:
            env.close()

        for p in self.process_list:
            p.join()

        del(self.process_list)

    def step_with_logprobs(self):

        loaded = False

        while(self.stop_collecting.item() == 0):

            if not loaded:
                if self.episode_count % self.agent_update_episodes == 0 and self.episode_count != 0:
                    loaded = True
                    self.agent.load(self.agent_dir)
                    time.sleep(0.3)

            if self.episode_count % self.agent_update_episodes != 0:
                loaded = False

            if self.step_control.sum().item() == self.num_instances:
                observations = self.observations
                action_probs = self.agent.actor(observations.to(self.device))
                actions = torch.tensor([self.agent.get_action(obs.to(self.device)) for obs in observations])
                if self.action_space == 'discrete':
                    actions_temp = torch.argmax(actions, dim=-1).long().to(self.device)
                    logprobs = Categorical(action_probs).log_prob(actions_temp).to(self.device)
                self.observations[:,0:self.acs_dim] = actions

                self.log_probs[:,0] = logprobs
                self.step_control.fill_(0)

            if self.episode_count == self.total_episodes:
                self.stop_collecting[0] = 1


        for env in self.envs:
            env.close()

        for p in self.process_list:
            p.join()

        del(self.process_list)


    def launch_envs(self):

        environment = load_class('shiva.envs', self.configs['sub_type'])

        if self.configs['sub_type'] == 'RoboCupEnvironment':
            self.envs.append(environment(self.configs, 9853))
            self.acs_dim = self.envs[0].action_space['acs_space']
            self.obs_dim = self.envs[0].observation_space
            self.envs[0].close()

        else:
            for i in range(self.num_instances):
                self.configs['seed'] = i
                self.envs.append(environment(self.configs))
            self.acs_dim = self.envs[0].action_space['acs_space']
            self.obs_dim = self.envs[0].observation_space
            
        #Shared tensor will be used for communication between environment wrapper process and individual environment processes

        # print(self.discrete)
        self.observations = torch.zeros(self.num_instances, max(self.obs_dim, self.acs_dim)).share_memory_()
        #Shared tensor will let control data flow through the tensor
        # 0 signals env to process, 1 signals multi wrapper to process
        self.step_control = torch.zeros(self.num_instances).share_memory_()
        #Shared tensor will signal to the envs when to stop collecting episodes
        self.stop_collecting = torch.zeros(1).share_memory_()
        self.done_count = 0
        if self.logprobs:
            self.log_probs = torch.zeros(self.num_instances, 1).share_memory_()
            self.process_list = launch_processes(self.envs, self.observations, self.action_available, self.step_count, self.step_control, self.stop_collecting, self.waitForLearner, self.queue, self.max_episode_length, logprobs=self.log_probs, num_instances=self.num_instances)
            self.step_with_logprobs()
        else:
            if self.configs['sub_type'] == 'RoboCupEnvironment':
                self.process_list = launch_robo_process(self.observations,self.action_available,self.step_count, self.step_control, self.stop_collecting,self.waitForLearner,self.queue, self.max_episode_length, self.configs ,num_instances=self.num_instances)
                self.step_without_log_probs()
            else:
                self.process_list = launch_processes(self.envs, self.observations,self.action_available,self.step_count, self.step_control, self.stop_collecting,self.waitForLearner,self.queue, self.max_episode_length,num_instances=self.num_instances)
                self.step_without_log_probs()

def launch_processes(envs, observations, action_available, step_count, step_control, stop_collecting,waitForLearner,queue, max_episode_length,logprobs = None,num_instances=1):

    process_list = []

    for i in range(num_instances):
        if logprobs is not None:
            p = mp.Process(target = process_target_with_log_probs, args=(envs[i],observations,step_count,step_control,stop_collecting,i,queue,logprobs, max_episode_length,) )
        else:
            p = mp.Process(target = process_target, args=(envs[i],observations, action_available,step_count,step_control,stop_collecting,waitForLearner,i,queue,max_episode_length,) )
        p.start()
        process_list.append(p)

    return process_list


def launch_robo_process( observations, action_available, step_count, step_control, stop_collecting,waitForLearner,queue, max_episode_length,configs,logprobs = None,num_instances=1):

    process_list = []

    for i in range(num_instances):
        if logprobs is not None:
            p = mp.Process(target = process_target_with_log_probs, args=(envs[i],observations,step_count,step_control,stop_collecting,i,queue,logprobs, max_episode_length,) )
        else:
            p = mp.Process(target = robo_process_target, args=(observations, action_available,step_count,step_control,stop_collecting,waitForLearner, configs,i,queue,max_episode_length,) )
        p.start()
        process_list.append(p)

    return process_list

def process_target(env,observations,action_available,step_count,step_control,stop_collecting, waitForLearner, id, queue,max_ep_length):

    observation_space = env.observation_space
    action_space = env.action_space['acs_space']
    ep_observations = np.zeros((max_ep_length,observation_space))
    ep_actions= np.zeros((max_ep_length,action_space))
    ep_rewards= np.zeros((max_ep_length,1))
    ep_next_observations= np.zeros((max_ep_length,observation_space))
    ep_dones= np.zeros((max_ep_length,1))
    idx = 0
    env.reset()
    observation = env.get_observation()
    observations[id] = torch.tensor(observation).float()
    step_control[id] = 1

    while(stop_collecting.item() == 0):
        if step_control[id] == 0 and waitForLearner.item() == 0:
            time.sleep(0.06)
            action = observations[id][:action_space].numpy()
            action_available[0] = 0
            next_observation, reward, done, more_data = env.step(action, discrete_select='sample')
            ep_observations[idx] = observation
            ep_actions[idx] = more_data['action']
            ep_rewards[idx] = reward
            ep_next_observations[idx] = next_observation
            ep_dones[idx] = int(done)
            idx += 1
            step_count +=1

            if done:
                exp = copy.deepcopy(
                            zip(
                                ep_observations[:idx],
                                ep_actions[:idx],
                                ep_rewards[:idx],
                                ep_next_observations[:idx],
                                ep_dones[:idx]
                            )
                        )
                queue.put(exp)
                env.reset()
                observation = env.get_observation()
                observations[id] = observation
                ep_observations.fill(0)
                ep_actions.fill(0)
                ep_rewards.fill(0)
                ep_next_observations.fill(0)
                ep_dones.fill(0)
                idx = 0
                step_control[id] = 1
            else:
                observations[id] = torch.from_numpy(next_observation)
                observation = next_observation
                step_control[id] = 1


def process_target_with_log_probs(env,observations,step_count,step_control,stop_collecting, id, queue,logprobs, max_ep_length):
# def process_target(self):
    #tensor for storing episodes per process/env
    observation_space = env.observation_space
    action_space = env.action_space['acs_space']
    ep_observations = np.zeros((max_ep_length,observation_space))
    ep_actions = np.zeros((max_ep_length,action_space))
    ep_rewards = np.zeros((max_ep_length,1))
    ep_logprobs = np.zeros((max_ep_length, action_space))
    ep_next_observations = np.zeros((max_ep_length,observation_space))
    ep_dones = np.zeros((max_ep_length,1))
    idx = 0
    env.reset()
    observation = env.get_observation()
    observations[id] = torch.tensor(observation).float()
    step_control[id] = 1

    while(stop_collecting.item() == 0):
        #print('Hello')
        #print('Process: ', step_control[id])
        if(step_control[id] == 0):
            action = observations[id][:action_space].numpy()
            log_probs = logprobs[id][0].numpy()
            next_observation, reward, done, more_data = env.step(action)
            ep_observations[idx] = observation
            ep_actions[idx] = action
            ep_rewards[idx] = reward
            ep_logprobs[idx] = log_probs
            ep_next_observations[idx] = next_observation
            ep_dones[idx] = int(done)
            idx += 1
            step_count +=1

            # print("Hello")

            '''t = [observation, action, reward, next_observation, int(done)]
            exp = copy.deepcopy(t)
            print('List: ',exp)
            print('Tensor: ', torch.tensor(exp))
            episode_trajectory[episode_index:episode_index + len(exp)] = torch.tensor(exp)
            episode_index += len(exp)'''
            if done:
                exp = copy.deepcopy(
                            zip(
                                ep_observations[:idx],
                                ep_actions[:idx],
                                ep_rewards[:idx].tolist(),
                                ep_logprobs[:idx,0],
                                ep_next_observations[:idx],
                                ep_dones[:idx]
                                )
                            )
                queue.put(exp)
                env.reset()
                observation = env.get_observation()
                observations[id] = observation
                ep_observations.fill(0)
                ep_actions.fill(0)
                ep_rewards.fill(0)
                ep_logprobs.fill(0)
                ep_next_observations.fill(0)
                ep_dones.fill(0)
                idx = 0
                step_control[id] = 1
            else:
                observations[id] = torch.from_numpy(next_observation)
                observation = next_observation
                step_control[id] = 1


def robo_process_target(observations,action_available,step_count,step_control,stop_collecting,waitForLearner,config,id,queue,max_ep_length):
    
    config['seed'] = id
    env = load_class('shiva.envs', config['sub_type'])
    env = env(config, (id+45) * 1000)

    observation_space = env.observation_space
    action_space = env.action_space['acs_space']
    ep_observations = np.zeros((max_ep_length,observation_space))
    ep_actions= np.zeros((max_ep_length,action_space))
    ep_rewards= np.zeros((max_ep_length,1))
    ep_next_observations= np.zeros((max_ep_length,observation_space))
    ep_dones= np.zeros((max_ep_length,1))
    idx = 0
    env.reset()
    observation = env.get_observation()
    observations[id][:observation_space] = torch.tensor(observation).float()
    step_control[id] = 1

    while(stop_collecting.item() == 0):
        if step_control[id] == 0 and waitForLearner.item() == 0:
            time.sleep(0.001)
            action = observations[id][:action_space].numpy()
            action_available[0] = 0
            next_observation, reward, done, more_data = env.step(torch.tensor(action), discrete_select='sample')
            ep_observations[idx] = observation
            ep_actions[idx] = more_data['action']
            ep_rewards[idx] = reward
            ep_next_observations[idx] = next_observation
            ep_dones[idx] = int(done)
            idx += 1
            step_count +=1

            if done:
                exp = copy.deepcopy(
                            zip(
                                ep_observations[:idx],
                                ep_actions[:idx],
                                ep_rewards[:idx],
                                ep_next_observations[:idx],
                                ep_dones[:idx]
                            )
                        )
                queue.put(exp)
                env.reset()
                observation = env.get_observation()
                observations[id][:observation_space] = torch.tensor(observation)
                ep_observations.fill(0)
                ep_actions.fill(0)
                ep_rewards.fill(0)
                ep_next_observations.fill(0)
                ep_dones.fill(0)
                idx = 0
                step_control[id] = 1
            else:
                observations[id][:observation_space] = torch.from_numpy(next_observation)
                observation = next_observation
                step_control[id] = 1
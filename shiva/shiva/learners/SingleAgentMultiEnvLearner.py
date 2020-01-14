# from shiva.core.admin import Admin

from shiva.learners.Learner import Learner
from shiva.helpers.config_handler import load_class
import helpers.misc as misc
import torch.multiprocessing as mp
import os
import torch
import copy
import random
import time
import numpy as np



class SingleAgentMultiEnvLearner(Learner):
    '''
        Work in progress.
        One MultiEnv Learner for all algorithms

    '''

    def __init__(self, learner_id, config):
        super(SingleAgentMultiEnvLearner,self).__init__(learner_id, config)
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.aggregator_index = torch.zeros(1).share_memory_()
        self.tot_rew_idx = torch.zeros(1).share_memory_()
        self.saveLoadFlag = torch.zeros(1).share_memory_()
        self.ep_count = torch.zeros(1).share_memory_()
        self.step_count = torch.zeros(1).share_memory_()
        self.updates = 0
        self.agent_dir = os.getcwd() + self.agent_path
        self.waitForLearner = torch.zeros(1).share_memory_()
        self.MULTI_ENV_FLAG = True

    def run(self):

        while self.ep_count < self.episodes:
            # while not self.queue.empty():

                # this should prevent the GymWrapper from getting to far ahead of the learner
                # Makes the gym wrapper stop collecting momentarily
                # if self.queue.qsize() >= 5:
                #     self.waitForLearner[0] = 
                    
            # time.sleep(0.06)

            # temp = self.ep_count

            '''
            
            Collect From Aggregator and
            '''

            if self.aggregator_index.item():

                idx = int(self.aggregator_index.item())
                t = int(self.tot_rew_idx.item())

                exp = copy.deepcopy(
                    [
                        self.obs_buffer[:idx],
                        self.acs_buffer[:idx],
                        self.rew_buffer[:idx],
                        self.next_obs_buffer[:idx],
                        self.done_buffer[:idx]
                    ]
                )

                for i in range(t):
                    self.reward_per_episode = self.ep_metrics_buffer[t,0].item()
                    self.steps_per_episode = int(self.ep_metrics_buffer[t,1].item())
                    self.collect_metrics(episodic=True)

                if len(self.buffer) <= self.buffer.batch_size:
                    self.collect_metrics(episodic=True)

                self.aggregator_index[0] = 0
                self.tot_rew_idx[0] = 0
                self.buffer.push(exp)



            if self.configs['Algorithm']['algorithm'] == 'PPO':
                observations, actions, rewards, logprobs, next_observations, dones = zip(*exp)
                print("Episode {} Episodic Reward {} ".format(self.ep_count.item(), np.array(rewards).sum()))
                exp = [
                        torch.tensor(observations),
                        torch.tensor(actions),
                        torch.tensor(rewards),
                        torch.tensor(next_observations),
                        torch.tensor(dones),
                        torch.tensor(logprobs)
                ]
                self.step_count += len(observations)
                for i in range(len(observations)):
                    self.reward_per_step = rewards[i][0]
                    self.collect_metrics(episodic=False)
                    # self.step_count += 1
                self.buffer.push(exp)
                if self.buffer.current_index - 1 >= self.update_episodes:
                    self.alg.update(self.agent,self.buffer,self.step_count)
                self.collect_metrics(episodic=False)

            if len(self.buffer) > self.buffer.batch_size:
                # start_time = time.time()
                self.alg.update(self.agent,self.buffer,self.step_count, episodic=True)
                # print("Update--- %s seconds ---" % (time.time() - start_time))
                # print("this happened")
                self.collect_metrics(episodic=True)


                # self.alg.update(self.agent,self.buffer,self.step_count, episodic=True)
                # self.ep_count += 1

                # if self.ep_count.item() / self.configs['Algorithm']['update_episodes'] >= self.updates:
                    # self.alg.update(self.agent,self.buffer,self.step_count, episodic=True)
                    # print("hello")

            if self.saveLoadFlag.item() == 0:
                # print("Multi Learner:",self.agent_dir)
                # self.agent.save_agent(self.agent_dir,self.step_count)
                self.agent.save(self.agent_dir, self.step_count.item())
                print("Agent was saved")
                self.saveLoadFlag[0] = 1

                # self.updates += 1


        # self.p.join()
        #print('Hello')
        # del(self.p)
        del(self.queue)


    def step(self):

        observation = self.env.get_observation()

        action = self.agent.get_action(observation)

        next_observation, reward, done, more_data = self.env.step(action)

        """Temporary fix for Unity as it receives multiple observations"""
        if len(observation.shape) > 1:
            z = copy.deepcopy(zip(observation, action, reward, next_observation, done))
            for obs, act, rew, next_obs, don in z:
                exp = [obs, act, rew, next_obs, int(don)]
                exp = copy.deepcopy(exp)
                self.buffer.append(exp)
        else:
            t = [observation, action, reward, next_observation, int(done)]
            deep = copy.deepcopy(t)
            self.buffer.append(deep)
        """"""

    def create_environment(self):
        # create the environment and get the action and observation spaces
        environment = load_class('shiva.envs', self.configs['Environment']['type'])
        return environment(self.configs['Environment'],self.queue,self.agent,self.ep_count,self.agent_dir,self.episodes, self.saveLoadFlag, self.waitForLearner, self.step_count)

    def create_algorithm(self):
        if self.configs['Environment']['sub_type'] == 'RoboCupEnvironment':
            algorithm = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])
        else:
            algorithm = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
            acs_continuous = self.env.action_space_continuous
            acs_discrete= self.env.action_space_discrete
            return algorithm(self.env.get_observation_space(), self.env.get_action_space(), [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self, obs_dim, ac_dim):
        buffer = load_class('shiva.buffers',self.configs['Buffer']['type'])
        return buffer(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.env.num_instances, obs_dim, ac_dim)

    def create_aggregator(self, obs_dim, acs_dim):

        self.aggregator = mp.Process(
                                        target = data_aggregator,

                                        args = (
                                            self.obs_buffer,
                                            self.acs_buffer,
                                            self.rew_buffer,
                                            self.next_obs_buffer,
                                            self.done_buffer,
                                            self.ep_metrics_buffer,
                                            self.queue,
                                            self.aggregator_index,
                                            self.tot_rew_idx,
                                            self.ep_count,
                                            self.aggregator_size,
                                            obs_dim,
                                            acs_dim,
                                        )
                                    )

        self.aggregator.start()

    def get_agents(self):
        return self.agents

    def get_algorithm(self):
        return self.alg

    def launch(self):
        environment = load_class('shiva.envs', self.configs['Environment']['sub_type'])
        self.configs['Environment']['port'] = 20000
        self.env = environment(self.configs)
        obs_dim = self.env.get_observation_space()
        acs_dim = self.env.get_action_space()['acs_space']
        time.sleep(5)
        self.env.close()



        # if buffer set to true in config
        if self.using_buffer:
            # Tensor replay buffer at the moment
            self.buffer = self.create_buffer(obs_dim, acs_dim)

        self.obs_buffer = torch.zeros((10_000, obs_dim), requires_grad=False).share_memory_()
        self.acs_buffer = torch.zeros( (10_000, acs_dim) ,requires_grad=False).share_memory_()
        self.rew_buffer = torch.zeros((10_000, 1),requires_grad=False).share_memory_()
        self.next_obs_buffer = torch.zeros((10_000, obs_dim),requires_grad=False).share_memory_()
        self.done_buffer = torch.zeros((10_000, 1),requires_grad=False).share_memory_()
        self.ep_metrics_buffer = torch.zeros((10_000, 2),requires_grad=False).share_memory_()


        self.create_aggregator(obs_dim, acs_dim)


        # Launch the algorithm which will handle the
        self.alg = self.create_algorithm()
        # Create the agent
        if self.load_agents:
            self.agent= self.load_agent(self.load_agent)
        else:
            self.agent = self.alg.create_agent()
            self.agent.save(self.agent_dir,self.step_count.item())
            print("first agent saved to directory")

        # Launch the environment
        self.env = self.create_environment()

        print('Launch Successful.')


    def save_agent(self):
        pass

    def load_agent(self, path):
        return shiva._load_agents(path)[0]

    def get_metrics(self, episodic=False):
        if not episodic:
            metrics = [
                # ('Reward/Per_Step', self.reward_per_step)
            ]
        else:
            metrics = [
                ('Reward/Per_Episode', self.reward_per_episode),
                ('Agent/Steps_Per_Episode', self.steps_per_episode)
            ]

            # print("Episode {} complete. Total Reward: {}".format(self.done_count, self.reward_per_episode))

        return metrics


def data_aggregator(obs_buffer, acs_buffer, rew_buffer, next_obs_buffer,done_buffer, ep_metrics_buffer, queue, current_index, tot_rew_idx, ep_count, max_size, obs_dim, acs_dim):

    while True:

        time.sleep(0.5)

        while not queue.empty():

            exps = queue.get()
            # print(queue.qsize())
            ep_count += 1
            obs, ac, rew, next_obs, done = zip(*exps)

            obs = torch.tensor(obs)
            ac = torch.tensor(ac)
            rew = torch.tensor(rew).reshape(-1,1)
            next_obs = torch.tensor(next_obs)
            done = torch.tensor(done).reshape(-1,1)


            '''
                Collect metrics here
                average reward
                sum of rewards

            '''
            nentries = len(obs)

            avg_rew = rew.mean()
            tot_rew = rew.sum()

            print("Episode {} Episodic Reward {} ".format(int(ep_count.item()), tot_rew))

            idx = int(current_index.item())
            t = int(tot_rew_idx.item())

            obs_buffer[idx:idx+nentries] = obs
            acs_buffer[idx:idx+nentries] = ac
            rew_buffer[idx:idx+nentries] = rew
            next_obs_buffer[idx:idx+nentries] = next_obs
            done_buffer[idx:idx+nentries] = done
            ep_metrics_buffer[t:t+1, 0] = tot_rew
            ep_metrics_buffer[t:t+1, 1] = nentries
            tot_rew_idx += 1
            current_index += nentries


    # def close(self):

        # for env in self.env.envs:
            # env.close()

        # for p in self.env.process_list:
        #     p.close()

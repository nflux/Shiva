import time, os, copy
import torch
import torch.multiprocessing as mp
import numpy as np
from mpi4py import MPI

from shiva.core.admin import Admin

from shiva.metalearners.CommMultiLearnerMetaLearnerServer import get_meta_stub
from shiva.helpers.launch_servers_helper import start_learner_server
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub

from shiva.helpers.config_handler import load_class

class CommMultiAgentLearner():
    def __init__(self, id):
        self.id = id
        self.address = ':'.join(['localhost', '50000'])
        self.agents = []

    def launch(self, meta_address):
        self.meta_stub = get_meta_stub(meta_address)
        self.debug("gRPC Request Meta for Configs")
        self.configs = self.meta_stub.get_configs()
        self.debug("gRPC Received Config from Meta")

        # self.meta_comm = MPI.Comm.Get_parent()
        # self.id = self.meta_comm.Get_rank()
        # self.configs = {}
        # self.meta_comm.bcast(self.configs, root=0) # receive configs
        # self.debug("MPI Received Config from Meta")

        {setattr(self, k, v) for k, v in self.configs['Learner'].items()}
        Admin.init(self.configs['Admin'])
        Admin.add_learner_profile(self, function_only=True)

        # initiate server
        self.comm_learner_server, self.learner_tags = start_learner_server(self.address, maxprocs=1)
        time.sleep(1)
        # self.debug("MPI send Configs to LearnerServer")
        self.comm_learner_server.send(self.configs, 0, self.learner_tags.configs)

        self.debug("gRPC send LearnerSpecs to MetaServer")
        self.meta_stub.send_learner_specs(self._get_learner_specs())

        # receive multienv specs
        self.menv_specs = self.comm_learner_server.recv(None, 0, self.learner_tags.menv_specs)
        self.menv_stub = get_menv_stub(self.menv_specs['address'])

        self.num_agents = 1 # this is given by the environment or by the metalearner

        self.debug("Ready to instantiate algorithm, buffer and agents!")

        self.queue = mp.Queue(maxsize=self.queue_size)
        self.aggregator_index = torch.zeros(1).share_memory_()
        self.metrics_idx = torch.zeros(1).share_memory_()
        # self.agent_dir = os.getcwd() + self.agent_path

        self.observation_space = self.menv_specs['env_specs']['observation_space']
        self.action_space = self.menv_specs['env_specs']['action_space']

        self.alg = self.create_algorithm(self.observation_space, self.action_space)

        self.buffer = self.create_buffer()

        # buffers for the aggregator
        self.obs_buffer = torch.zeros((self.aggregator_size, self.observation_space), requires_grad=False).share_memory_()
        self.acs_buffer = torch.zeros((self.aggregator_size, self.action_space), requires_grad=False).share_memory_()
        self.rew_buffer = torch.zeros((self.aggregator_size, 1), requires_grad=False).share_memory_()
        self.next_obs_buffer = torch.zeros((self.aggregator_size, self.observation_space), requires_grad=False).share_memory_()
        self.done_buffer = torch.zeros((self.aggregator_size, 1), requires_grad=False).share_memory_()
        self.ep_metrics_buffer = torch.zeros((self.aggregator_size, 2), requires_grad=False).share_memory_()

        self.create_aggregator(self.observation_space, self.action_space)

        if self.load_agents:
            self.agents = Admin._load_agents(self.load_agents, absolute_path=False)
        else:
            self.agents = [self.alg.create_agent(ix) for ix in range(self.num_agents)]

        self.agents[0].step_count = 0

        Admin.checkpoint(self, checkpoint_num=0, function_only=True)

        self.debug("Algorithm, Buffer and Agent created")
        self.menv_stub.send_learner_specs(self._get_learner_specs())
        self.debug("Learner Specs sent to MultiEnv")

        self.run()

    def run(self):
        self.step_count = 0
        self.done_count = 0
        trajectory = None
        while True:
            if self.aggregator_index.item():
                # trajectory = self.comm_learner_server.recv(None, 0, self.learner_tags.trajectories) # blocking communication

                idx = int(self.aggregator_index.item())
                t = int(self.metrics_idx.item())

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
                    self.ep_count += 1
                    self.reward_per_episode = self.ep_metrics_buffer[i, 0].item()
                    self.steps_per_episode = int(self.ep_metrics_buffer[i, 1].item())
                    self._collect_metrics(episodic=True)

                self.aggregator_index[0] = 0
                self.metrics_idx[0] = 0
                self.buffer.push(exp)

            # #############################
            # ## old without aggregator
            # trajectory = self.comm_learner_server.recv(None, 0, self.learner_tags.trajectories) # blocking communication
            #
            # self.done_count += 1
            # for observation, action, reward, next_observation, done in trajectory:
            #     exp = list(map(torch.clone, (torch.tensor(observation), torch.tensor(action), torch.tensor(reward), torch.tensor(next_observation), torch.tensor([done], dtype=torch.bool))))
            #     self.buffer.push(exp)
            #     self.step_count += 1
            #
            # if self.done_count % self.save_checkpoint_episodes == 0:
            #     self.alg.update(self.agents[0], self.buffer, self.done_count, episodic=True)
            #     self.agents[0].step_count = self.step_count
            #     # self.debug("Sending Agent Step # {}".format(self.step_count))
            #     Admin.checkpoint(self, checkpoint_num=self.done_count, function_only=True)
            #     self._collect_metrics()
            #     self.menv_stub.send_new_agents(Admin.get_last_checkpoint(self))
            # # self.debug("Sent new agents")
            # ###############################

    def create_aggregator(self, obs_dim, acs_dim):

        self.aggregator = mp.Process(
            target=data_aggregator,

            args=(
                self.comm_learner_server, # send communication with the server to receive trajectory
                self.learner_tags.trajectories, # tag where the aggregator should be looking for incoming data from server
                self.obs_buffer,
                self.acs_buffer,
                self.rew_buffer,
                self.next_obs_buffer,
                self.done_buffer,
                self.ep_metrics_buffer,
                self.queue,
                self.aggregator_index,
                self.metrics_idx,
                self.ep_count,
                self.aggregator_size,
                obs_dim,
                acs_dim,
            )
        )

        self.aggregator.start()

    def create_algorithm(self, observation_space, action_space):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(self.menv_specs['env_specs']['observation_space'], self.menv_specs['env_specs']['action_space'], self.configs)

    def create_buffer(self):
        # SimpleBuffer
        # buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        # return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])
        # TensorBuffer
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['capacity'], self.configs['Buffer']['batch_size'], self.num_agents, self.menv_specs['env_specs']['observation_space'], self.menv_specs['env_specs']['action_space']['acs_space'])

    def _collect_metrics(self):
        metrics = self.alg.get_metrics(True)# + self.env.get_metrics(episodic)
        for metric_name, y_val in metrics:
            Admin.add_summary_writer(self, self.agents[0], metric_name, y_val, self.step_count)

    def _get_learner_specs(self):
        return {
            'id': self.id,
            'algorithm': self.configs['Algorithm']['type'],
            'address': self.address,
            'load_path': Admin.get_last_checkpoint(self),
            'num_agents': self.num_agents if hasattr(self, 'num_agents') else 0
        }


    def __getstate__(self):
        d = dict(self.__dict__)
        attributes_to_ignore = ['meta_stub', 'comm_learner_server', 'menv_stub', 'learner_tags']
        for a in attributes_to_ignore:
            try:
                del d[a]
            except:
                pass
        return d

    def debug(self, msg):
        print("PID {} Learner\t\t{}".format(os.getpid(), msg))


def data_aggregator(comm_server, tag, obs_buffer, acs_buffer, rew_buffer, next_obs_buffer,done_buffer, ep_metrics_buffer, queue, aggregator_index, metrics_idx, ep_count, max_size, obs_dim, acs_dim):

    while True:
        # time.sleep(0.06)
        # while not queue.empty():

        exps = comm_server.recv(None, 0, tag=tag)  # blocking communication

        # exps = queue.get()
        # print(queue.qsize())
        # ep_count += 1
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

        print("Episode {} Episodic Reward {} ".format(int(ep_count.item()), rew.sum().item()))

        idx = int(aggregator_index.item())
        t = int(metrics_idx.item())

        obs_buffer[idx:idx+nentries] = obs
        acs_buffer[idx:idx+nentries] = ac
        rew_buffer[idx:idx+nentries] = rew
        next_obs_buffer[idx:idx+nentries] = next_obs
        done_buffer[idx:idx+nentries] = done
        ep_metrics_buffer[t:t+1, 0] = rew.sum()
        ep_metrics_buffer[t:t+1, 1] = nentries
        metrics_idx += 1
        aggregator_index += nentries


    # def close(self):

        # for env in self.env.envs:
            # env.close()

        # for p in self.env.process_list:
        #     p.close()
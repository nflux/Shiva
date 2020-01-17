import time, os
import torch
import numpy as np

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

        self.debug("Ready to instantiate algorithm, buffer and agents!")

        self.alg = self.create_algorithm(self.menv_specs['env_specs']['observation_space'], self.menv_specs['env_specs']['action_space'])
        self.buffer = self.create_buffer()

        self.num_agents = 1
        self.agents = [self.alg.create_agent(ix) for ix in range(self.num_agents)]

        Admin.checkpoint(self, checkpoint_num=0, function_only=True)

        self.debug("Algorithm, Buffer and Agent created")
        self.menv_stub.send_learner_specs(self._get_learner_specs())
        self.debug("Learner Specs sent to MultiEnv")

        self.run()

    def run(self):
        step_count = 0
        done_count = 0
        trajectory = None
        while True:
            # self.debug("Waiting trajectory...")
            trajectory = self.comm_learner_server.recv(None, 0, self.learner_tags.trajectories) # blocking receive
            # self.debug("Ready to update: trajectory length {}".format(len(trajectory)))
            done_count += 1
            for exp in trajectory:
                self.buffer.append(exp)
                step_count += 1
            if done_count % self.save_checkpoint_episodes == 0:
                self.alg.update(self.agents[0], self.buffer, step_count)
                Admin.checkpoint(self, checkpoint_num=done_count, function_only=True)
                self.menv_stub.send_new_agents(Admin.get_last_checkpoint(self))
            # self.debug("Sent new agents")

    def create_algorithm(self, observation_space, action_space):
        algorithm_class = load_class('shiva.algorithms', self.configs['Algorithm']['type'])
        return algorithm_class(observation_space, action_space, [self.configs['Algorithm'], self.configs['Agent'], self.configs['Network']])

    def create_buffer(self):
        buffer_class = load_class('shiva.buffers', self.configs['Buffer']['type'])
        return buffer_class(self.configs['Buffer']['batch_size'], self.configs['Buffer']['capacity'])

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

def collect_forever(minibuffer, queue, ix, num_agents, acs_dim, obs_dim, metrics):
    '''
        Separate process who collects the data from the server queue and puts it into a temporary minibuffer
    '''
    while True: # maybe a lock here
        pass
        # trajectory = from_TrajectoryProto_2_trajectory( queue.pop() )
        # collect metrics
        # metrics['rewards'] =
        # push to minibuffer
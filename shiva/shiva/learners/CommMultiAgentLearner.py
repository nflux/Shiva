import time
import torch
import torch.multiprocessing as mp

from shiva.learners.Learner import Learner
from shiva.buffers.TensorBuffer import TensorBuffer

from shiva.metalearners.CommMultiLearnerMetaLearnerServer import get_meta_stub
from shiva.learners.CommMultiAgentLearnerServer import start_learner_server
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub

class CommMultiAgentLearner(Learner):
    _ADDRESS = 'localhost:50051'
    _SERVER_TRAJECTORY_QUEUE_SIZE = 1000
    meta_stub = None
    menv_stub = None

    def __init__(self, id, configs, meta_address):
        super(CommMultiAgentLearner, self).__init__(id, configs)

        # setup shared memory with the local server
        manager = mp.Manager()
        self.shared_dict = manager.dict()
        self.shared_dict['menv_specs'] = None
        self.shared_dict['evol_config'] = None
        self.shared_dict['trajectories_queue'] = mp.Queue(self._SERVER_TRAJECTORY_QUEUE_SIZE)
        # initiate server
        self.server_p = mp.Process(target=start_learner_server, args=(self._ADDRESS, self.shared_dict))
        # time.sleep(1) # wait for the server to start!

        # create stubs
        self.meta_stub = get_meta_stub(meta_address)
        while self.shared_dict['menv_specs'] is None: # wait for the EnvSpecs to arrive
            pass
        self.menv_stub = get_menv_stub(self.shared_dict['menv_specs']['address'])

        # instantiate algorithms & buffer
        self.alg = self.create_algorithms()
        self.buffer = self.create_buffer() # main replay buffer
        self.minibuffer = torch.zeros().share_memory_()
        self.metrics = manager.dict()
        { self.metrics[agent_id] for agent_id in self.agent_ids }
        self.minibuffer_p = mp.Process(target=collect_forever, args=(self.minibuffer, self.shared_dict['trajectory_queue'], 0, self.num_agents, self.action_dim, self.observation_dim, self.metrics))

        self.run()

    def run(self):
        # send agents to the multi environment
        self.send_new_agents_to_menv()
        time.sleep(1) # wait a bit for environments to start and trajectories to accumulate
        while True:
            self.buffer.merge(self.minibuffer.get_all())
            self.alg.update()
            self.meta_stub.send_train_metrics(self.metrics)
            self.send_new_agents_to_menv()

    def send_new_agents_to_menv(self):
        '''
            Sends the New Agents information to the MultiEnv
        '''
        self.menv_stub.send_new_agents(self.agents)

def collect_forever(minibuffer, queue, ix, num_agents, acs_dim, obs_dim, metrics):
    '''
        Separate process who collects the data from the server queue and puts it into a temporary minibuffer
    '''
    while True: # maybe a lock here
        trajectory = from_TrajectoryProto_2_trajectory( queue.pop() )
        # collect metrics
        # metrics['rewards'] =
        # push to minibuffer
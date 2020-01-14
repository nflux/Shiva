import time
import torch.multiprocessing as mp

from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

from shiva.metalearners.CommMultiLearnerMetaLearnerServer import start_meta_server
from shiva.learners.CommMultiAgentLearner import CommMultiAgentLearner
from shiva.learners.CommMultiAgentLearnerServer import get_learner_stub
from shiva.envs.CommMultiEnvironmentServer import get_menv_stub
from shiva.envs.CommMultiEnvironment import CommMultiEnvironment

from shiva.core.communication_objects.configs_pb2 import StatusType # only thing being used for gRPC specific

class CommMultiLearnerMetaLearner(MetaLearner):
    _ADDRESS = "localhost:50155"
    learners_stub = None

    def __init__(self, configs):
        super(CommMultiLearnerMetaLearner, self).__init__(configs)
        self.configs = configs

        # setup shared memory with the local server
        manager = mp.Manager()
        self.shared_dict = manager.dict()
        self.shared_dict['learners_info'] = None
        self.shared_dict['menvs_info'] = None
        self.shared_dict['train_metrics'] = None
        self.shared_dict['eval_config'] = None

        # initiate server
        self.meta_server_p = mp.Process(target=start_meta_server, args=(self._ADDRESS, self.shared_dict))
        self.meta_server_p.start()
        # time.sleep(1) # wait for the server to start!

        '''
            Assuming 1 MultiEnv for now!
            Need to think how the Config would look for all scenarios of decentralized/centralized learners
        '''
        # initiate multienv processes
        self.menv_p = mp.Process(target=CommMultiEnvironment, args=(0, self.configs['Environment'], self._ADDRESS))
        self.menv_p.start()
        # time.sleep(1) # wait for the server to start!

        # initiate learners processes
        for ix in range(self.num_learners):
            learner_config = self.configs['Learner'] # for now, 1 config for all learners
            learner_p = mp.Process(target=CommMultiAgentLearner, args=(ix, learner_config, self._ADDRESS))
            learner_p.start()
        time.sleep(1)  # wait for the learners to start

        # check all learners have checked-in & create stub to communicate with them
        learners_checkin = 0
        while learners_checkin < self.num_learners:
            for learner_id, info in self.shared_dict['learners_info'].items():
                if info.status == StatusType.RUN:
                    learners_checkin += 1
                    self.learners_stub[learner_id] = get_learner_stub(info['address'])
            time.sleep(1)

        # check all menv have checked-in (& optionally create stub to communicate with them)
        menvs_checkin = 0
        while menvs_checkin < self.num_multienvs:
            for menv_id, info in self.shared_dict['menvs_info'].items():
                if info.status == StatusType.RUN:
                    menvs_checkin += 1
                    # self.shared_dict['menv_info'][menv_id]['stub'] = get_menv_stub(info['address'])

        # distribute the menv specs to the specific learners
        '''
            Only 1 menv for all the learners (centralized approach)
        '''
        for learner_id, info in self.shared_dict['learners_info'].items():
            this_learner_menv_specs = self.shared_dict['menv_specs'][learner_id]
            self.learners_stub[learner_id].send_menv_specs(this_learner_menv_specs)

    def run(self):
        while True:
            # do something with TrainingMetrics from learners?

            # check evaluation metrics and distribute evolution configs to learners

    def close(self):
        print('Cleaning up!')
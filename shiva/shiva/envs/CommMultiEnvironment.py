import time
import torch.multiprocessing as mp

from shiva.metalearners.CommMultiLearnerMetaLearner import get_meta_stub
from shiva.envs.CommMultiEnvironmentServer import serve_multienv
from shiva.envs.CommEnvironment import create_comm_env


class CommMultiEnvironment():
    _ADDRESS = "localhost:56000"
    meta_stub = None

    def __init__(self, menv_id, configs, meta_address):
        {setattr(self, k, v) for k, v in configs['Environment'].items()}
        self.id = menv_id
        self.configs = configs
        self.meta_stub = get_meta_stub(meta_address)

        # setup shared memory with the local server
        manager = mp.Manager()
        self.shared_dict = manager.dict()
        self.shared_dict['agents'] = None
        self.shared_dict['learners_info'] = None
        self.shared_dict['env_specs'] = None
        # initiate server
        self.menv_server_p = mp.Process(target=serve_multienv, args=(self._ADDRESS, self.shared_dict))
        self.menv_server_p.start()
        time.sleep(1) # wait for the server to start!

        # initiate individual envs
        self.env_p = []
        env_cls = self.configs['Environment']['type']  # all envs with same config
        for ix in range(self.num_instances):
            send_specs_to_menv = True if ix == 0 else False # one time message to get the environment specs
            p = mp.Process(target=create_comm_env, args=(env_cls, self.configs, self._ADDRESS, send_specs_to_menv))
            p.start()
            self.env_p.append(p)

        while self.shared_dict['env_specs'] is None:
            # wait for the env_specs from one environment to arrive
            pass

        #
        # do a handshake between menv and envs here? check everything is good
        #

        self.meta_stub.send_menv_specs(self._get_menv_specs())

        while self.shared_dict['learners_info'] is None: # menv waits for the learners info to pass to the individual envs
            pass


        learners_address = {}


    def _get_menv_specs(self):
        return {
            'menv_id': self.id,
            'env_specs': self.shared_dict['env_specs'],
            'num_envs': self.num_instances,
            'address': self._ADDDRESS
        }
from .robocup.rc_env import rc_env
from .Environment import Environment
from .robocup.HFO.bin import Communicator
import socket, time, pickle

class RoboCupDDPGEnvironment(Environment):
    def __init__(self, config):
        self.env = rc_env(config)
        self.env.launch()
        self.left_actions = self.env.left_actions
        self.left_params = self.env.left_action_params
        self.obs = self.env.left_obs
        self.rews = self.env.left_rewards
        self.world_status = self.env.world_status
        self.observation_space = self.env.left_features
        self.action_space = self.env.acs_dim
        self.step_count = 0
        self.render = self.env.config['env_render']
        self.done = self.env.d

        self.load_viewer()

        self._comm = Communicator.ClientCommunicator(port=6003, sock_type=socket.SOCK_STREAM)
        self._comm._sock.connect(('127.0.0.1', 6003))
        time.sleep(1)
        # self._comm._sock.bind

    def step(self, left_actions, left_params):
        self.left_actions = left_actions
        self.left_params = left_params
        self.obs,self.rews,_,_,self.done,_ = self.env.Step(left_actions=left_actions, left_params=left_params)

        # print(self._comm._addr)
        # self._comm.sendMsg('HelloWorld')
        while True:
            try:
                self._comm._sock.sendall(pickle.dumps(self.obs))
                print('hey')
                # self._comm._sock.sendall('(move (player ' + 'HELIOS_18_CLONE' + ' 11) -5 10 10 10 10)'.encode())
                while True:
                    msg = self._comm._sock.recv(1024)
                    print(msg)
                    if b'True' == msg:
                        break
                break
            except:
                time.sleep(0.0001)

        return self.obs, self.rews, self.done

    def get_observation(self):
        return self.obs

    def get_actions(self):
        return self.left_actions, self.left_params

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env._start_viewer()
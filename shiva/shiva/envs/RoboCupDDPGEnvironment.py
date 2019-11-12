from .robocup.rc_env import rc_env
from .Environment import Environment
from .robocup.HFO.bin import Communicator
# import socket, time, pickle
import socket, time

class RoboCupDDPGEnvironment(Environment):
    def __init__(self, config):
        super(RoboCupDDPGEnvironment, self).__init__(config)
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

        while True:
            try:
                self.comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.comm.connect(('127.0.0.1', self.imit_port))
                break
            except:
                time.sleep(0.001)
        # while True:
        #     self.comm.send(b"True")
        #     msg = self.comm.recv(8192)
        #     if msg == b"True":
        #         break

        # self._comm = Communicator.ClientCommunicator(port=2001)
        # self._comm.sendMsg('(init (version 8.0))')
        # self._comm.sendMsg('(ear on)')
        # self.f = open("demofile2.txt", "a")

        # self._comm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self._comm.settimeout(None)
        # self._comm.sendto(('(init (version 8.0))'+'\0').encode('utf-8'), ('localhost', 2001))
        # self._comm.connect(('127.0.0.1', 2001))
        # self._comm.listen()
        # self.conn, addr = self._comm.accept()
        # print('Accepted', addr)
        # input()
        # time.sleep(1)

    def step(self, left_actions, left_params):
        self.left_actions = left_actions
        self.left_params = left_params
        self.obs,self.rews,_,_,self.done,_ = self.env.Step(left_actions=left_actions, left_params=left_params)

        # msg = self._comm.recvMsg()
        # print(msg)
        # self._comm.sendMsg('(help)')
        # while True:
        #     self._comm.sendMsg('(help)')
        #     try:
        #         msg = self._comm.recvMsg()
        #     except:
        #         time.sleep(0.001)
        #     if msg == 'True':
        #         break
        
        # self.f.write("Timestep\n")
        while True:
            self.comm.send(b"True")
            msg = self.comm.recv(8192)
            if msg == b"True":
                break
                

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
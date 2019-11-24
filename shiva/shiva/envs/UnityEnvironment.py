import gym
from .Environment import Environment
import numpy as np
import torch
import socket
import time 

class UnityEnvironment(Environment):
    def __init__(self,environment):
        super(UnityEnvironment,self).__init__(environment)

        self.env = self.env_name
        # self.obs = self.reset()
        self.rews = 0
        self.world_status = False
        self.observation_space = self.set_observation_space()
        self.action_space = self.set_action_space()
        self.action_space_continuous = None
        self.action_space_discrete = None 
        self.step_count = 0

        # create a socket object 
        self.s = socket.socket()          
        print("Socket successfully created")
        # Bind to the port
        self.s.bind(('', self.port))
        print("socket binded to {}".format(self.port))
        # put the socket into listening mode 
        self.s.listen(10)      
        print("socket is listening")   

    def step(self,action):  
         
        # split the action
        action1 = bytes(str(action[0]) + " ", "utf-8")
        action2 = bytes(str(action[1]) + " ", "utf-8")

        # send back the actions
        self.clientSocket.send(action1)
        self.clientSocket.send(action2)

        # time.sleep(0.01)

        srd = self.clientSocket.recv(1024)

        # print("Unity Environment srd:", srd)

        # parse srd so we get the reward, done, and next state 
        srd = str(srd).strip('\'').split(' ')
        # print(srd)
        # print(srd[9])
        # this will require more string manipulation

        # not doing much with this currently
        # agent_id = srd[:1]

        self.next_observation = self.bytes2numpy(srd[1],srd[2], srd[3:6], srd[6:9])
        self.rews = np.float32(srd[9])
        self.world_status = np.bool(srd[10])

        self.step_count +=1

        if self.normalize:
            return self.next_observation, self.normalize_reward(), self.world_status, {'raw_reward': self.rews}
        else:
            return self.next_observation, self.rews, self.world_status, {'raw_reward': self.rews}

    def reset(self):
        pass

    def set_observation_space(self):

        self.observation_space = 4
        return 8

    def set_action_space(self):
        self.action_space = 2
        return 2

    def get_observation(self):
        # Establish connection with client. 
        self.clientSocket, self.address = self.s.accept()
        s = str(self.clientSocket.recv(1024)).strip('\'').split(' ')
        return self.bytes2numpy(s[1],s[2], s[3:6], s[6:9])

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        # Close the connection with the client
        self.clientSocket.close() 

    def bytes2numpy(self, a1, a2, p, v):
        # Turn everything into np floats and strip off string characters
        z = np.float32(a1)
        x = np.float32(a2)
        pos = np.array([np.float32( s.strip('(').strip(')').strip(',') ) for s in p ], dtype=np.float32)
        vel = np.array([np.float32( s.strip('(').strip(')').strip(',') ) for s in v ], dtype=np.float32)
        test = np.array([z, x, *list(pos), *list(vel)])
        return test

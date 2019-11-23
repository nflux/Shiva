import gym
from .Environment import Environment
import socket
import time 

class UnityEnvironment(Environment):
    def __init__(self,environment):
        super(UnityEnvironment,self).__init__(environment)

        self.env = self.env_name
        # self.obs = self.env.reset()
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
         
        # Establish connection with client. 
        clientSocket, address = self.s.accept()

        # action = [0.310956 ,0.4244231]

        # split the action
        action1 = bytes(str(action[0]) + " ", "utf-8")
        action2 = bytes(str(action[1]) + " ", "utf-8")

        time.sleep(10)

        # send back the actions
        clientSocket.send(action1)
        clientSocket.send(action2)


        srd = clientSocket.recv(1024)

        print(srd)

        # parse srd so we get the reward, done, and next state 
        srd = str(srd).split(' ')
        # this will require more string manipulation
        agent_id = srd[:1]
        next_state = srd[1:8]

        # good to go
        reward = float(srd[8:9])
        done = srd[9:]


        self.world_states = bool(done)



        self.obs, self.rews, self.world_status = self.env.step(action)
        self.step_count +=1
        self.load_viewer()

        if self.normalize:
            return self.obs, self.normalize_reward(), self.world_status, {'raw_reward': self.rews}
        else:
            return self.obs, self.rews, self.world_status, {'raw_reward': self.rews}

    def reset(self):
        self.obs = self.env.reset()

    def set_observation_space(self):

        self.observation_space = 4
        return 4
        # observation_space = 1
        # if self.env.observation_space.shape != ():
        #     for i in range(len(self.env.observation_space.shape)):
        #         observation_space *= self.env.observation_space.shape[i]
        # else:
        #     observation_space = self.env.observation_space.n

        # return observation_space

    def set_action_space(self):
        self.action_space = 2
        return 2
        # self.action_space_continuous = 2
        # if self.env.action_space.shape != ():
        #     for i in range(len(self.env.action_space.shape)):
        #         action_space *= self.env.action_space.shape[i]
        #     self.action_space_continuous = action_space
        # else:
        #     action_space = self.env.action_space.n
        #     self.action_space_discrete = action_space

    def get_observation(self):
        # Establish connection with client. 
        clientSocket, address = self.s.accept()
        # print('Got connection from', address )
        self.obs = clientSocket.recv(1024)
        return self.obs

    def get_action(self):
        return self.acs

    def get_reward(self):
        return self.rews

    def load_viewer(self):
        if self.render:
            self.env.render()

    def close(self):
        # Close the connection with the client
        clientSocket.close() 
        self.env.close()

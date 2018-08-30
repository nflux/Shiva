import itertools
import random
import numpy as np
import random 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.misc
import os 
import csv
import itertools 
import tensorflow.contrib.slim as slim
import numpy as np

import hfo
import time
import _thread as thread

 
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG


#matplotlib inline

#from helper import * 

# Engineered Reward Function
def getReward(s):    
      reward=0
      #--------------------------- 
      if s=='Goal':
        reward=1000
      #--------------------------- 
      elif s=='CapturedByDefense':
        reward=-1000
      #--------------------------- 
      elif s=='OutOfBounds':
        reward=-1000
      #--------------------------- 
      #Cause Unknown Do Nothing
      elif s=='OutOfTime':
        reward=-1000
      #--------------------------- 
      elif s=='InGame':
        reward=-10
      #--------------------------- 
      elif s=='SERVER_DOWN':  
        reward=0
      #--------------------------- 
      else:
        print("Error: Unknown GameState", s)
        reward = -1
      return reward


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def connect(self, feat_lvl, base, goalie, agent_ID):
    """ Connect threaded agent to server
    
    Args:
        feat_lvl: Feature level to use. ('high', 'low')
        base: Which base to launch agent to. ('left', 'right)
        goalie: Play goalie. (True, False)
        agent_ID: Integer representing agent index. (0-11)
        
    Returns:
        None, thread runs on server continually.
        
    
    """
 

    if feat_lvl == 'low':
        feat_lvl = hfo.LOW_LEVEL_FEATURE_SET
    elif feat_lvl == 'high':
        feat_lvl = hfo.HIGH_LEVEL_FEATURE_SET
        
    if base == 'left':
        base = 'base_left'
    elif base == 'right':
        base = 'base_right'        

    self.team_envs[agent_ID].connectToServer(feat_lvl,
                            config_dir='/home.sda4/home/ssajjadi/Desktop/ML/work/HFO-master/bin/teams/base/config/formations-dt', 
                        server_port=6000, server_addr='localhost', team_name=base, play_goalie=goalie)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    max_epLength = 100
    num_episodes = 20000
    

    # Once all agents have been loaded,
    # wait for action command, take action, update: obs, reward, and world status
    while(self.start):

        j = 0 # j to maximum episode length
        d = False
        self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # Get initial state

        while j < max_epLength:
            j+=1
            # If the action flag is set, each thread takes its action
            # Sets its personal flag to false, such that if all agents have taken their action 
            # and have updated values we may return to the Step function call
            # (This synchronizes all agents actions so that they halt until other agents have taken theirs)
            if(self.team_should_act_flag):
                self.team_envs[agent_ID].act(self.action_list[self.team_actions[agent_ID]]) # take the action
                self.world_status = self.team_envs[agent_ID].step() # update world
                self.team_rewards[agent_ID] = getReward(
                    self.team_envs[agent_ID].statusToString(self.world_status)) # update reward
                self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # update obs
                self.team_should_act[agent_ID] = False # Set personal action flag as done

                # Break if episode done
                if self.world_status == hfo.IN_GAME:
                    d = False
                else:
                    d = True

                if d == True:
                    break
                    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class HFO_env():
    """HFO_env() extends the HFO environment to allow for centralized execution.
    
    Attributes:
        num_TA (int): Number of teammate agents. (0-11)
        num_OA (int): Number of opponent agents. (0-11)
        team_actions (list): List contains the current timesteps action for each
            agent. Takes value between 0 - num_states and is converted to HFO action by
            action_list.
        action_list (list): Contains the mapping from numer action value to HFO action.
        team_should_act (list of bools): Contains a boolean flag for each agent. Is
            activated to be True by Step function and becomes false when agent acts.
        team_should_act_flag (bool): Boolean flag is True if agents have
            unperformed actions, becomes False if all agents have acted.
        team_obs (list): List containing obs for each agent.
        team_rewards (list): List containing reward for each agent
        start (bool): Once all agents have been launched, allows threads to listen for
            actions to take.
        world_states (list): Contains the status of the HFO world.
        team_envs (list of HFOEnvironment objects): Contains an HFOEnvironment object
            for each agent on team.
        opp_xxx attributes: Extend the same functionality for user controlled team
            to opposing team.
        
    Todo:
        * Functionality for synchronizing team actions with opponent team actions
        """
    
    def __init__(self, num_TA,num_OA,base,goalie,num_trials,fpt,feat_lvl,act_lvl):
        
        
        self.num_TA = num_TA
        self.num_OA = num_OA
        # Initialization of mutable lists to be passsed to threads
        self.team_actions = [3]*num_TA
        self.action_list = [hfo.DRIBBLE, hfo.SHOOT, hfo.REORIENT, hfo.GO_TO_BALL ]
        self.team_should_act = np.array([0]*num_TA)
        self.team_should_act_flag = False
        
        self.team_obs = [1]*num_TA
        self.team_rewards = [1]*num_TA

        self.opp_actions = [3]*num_OA
        self.opp_should_act = np.array([0]*num_OA)
        self.opp_should_act_flag = False

        self.opp_obs = [1]*num_OA
        self.opp_rewards = [1]*num_OA

        self.start = False
        
        self.world_status = 0

        # Create env for each teammate
        self.team_envs = [hfo.HFOEnvironment() for i in range(num_TA)]
        self.opp_team_envs = [hfo.HFOEnvironment() for i in range(num_OA)]


        # Create thread for each teammate
        for i in range(num_TA):
            print("Loading player %i" % i , "on team %s" % base)
            thread.start_new_thread(connect,(self, feat_lvl, base, 
                                             False,i,))
            time.sleep(1)

        # Create thread for each opponent (set one to goalie)
        if base == 'left':
            opp_base = 'right'
        elif base == 'right':
            opp_base = 'left'
            
        for i in range(num_OA):
            if i==0:
                thread.start_new_thread(connect,(self, feat_lvl, opp_base,
                                                 True,i,))
            else:
                thread.start_new_thread(connect,(self, feat_lvl, opp_base,
                                                 False,i,))
            time.sleep(1)
     
        
        print("All players loaded")
        print("Running synchronized threads")

        self.start = True
  
    def Observation(self,agent_id,side):
        """ Requests and returns observation from an agent from either team.
        
        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            side (str): Which team agent belongs to. ('team', 'opp')
        
        Returns:
            Observation from requested agent.
            
        """
        
        if side == 'team':
            return np.asarray(self.team_obs[agent_id])
        elif side == 'opp':
            return np.asarray(self.opp_obs[agent_id])
        
    def Reward(self,agent_id,side):
        """ Requests and returns reward from an agent from either team.
        
        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            side (str): Which team agent belongs to. ('team', 'opp')
        
        Returns:
            Reward from requested agent.
            
        """
        if side == 'team': 
            return self.team_rewards[agent_id]
        elif side == 'opp':
            return self.opp_rewards[agent_id]


    def Step(self,agent_id,action,side):
        """ Performs an action on agent and returns world_status from an agent from either team.
        
        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            side (str): Which team agent belongs to. ('team', 'opp')
        
        Returns:
            Status of HFO World.
                
        Todo:
            * Add halting until agents from both teams have received action instructions
            
        """
        if side == 'team':
            self.team_actions[agent_id] = action
        elif side == 'opp':
            self.opp_actions[agent_id] = action
        
        # Sets action flag to True allowing threads to take action only once each thread's personal action flag
        # is set to true.
        for i in range(len(self.team_should_act)):
            self.team_should_act[i] = True
            if self.team_should_act.all():
                self.team_should_act_flag = True
        # Halts until all threads have taken their action and resets flag to off
        while(self.team_should_act.any()):
            pass
        self.team_should_act_flag = False
      
        return self.world_status
    
   

  


env = HFO_env(1,0,'left',False,1000,1000,'high','high') 
  
for it in range(10):
    for ag in range(env.num_TA):
        env.Step(ag,random.randint(0,3),'team')
        print(env.team_obs[0])
        
  

maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                  adversary_alg= "MADDPG",
                                  tau=0.08,
                                  lr=0.0002,
                                  hidden_dim=64)

replay_buffer = ReplayBuffer(10000, 1,
                                 [12],
                                 [4])
t = 0
for ep_i in range(0, 1000):
        
        
        obs = np.asarray(env.team_obs)
         
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, 20 - ep_i) / 20
        maddpg.scale_noise(0 + (0.3 - 0.0) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(1000):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(1)]
            print('actions, ', actions)
            env.Step(actions, 0)
            #next_obs, rewards, dones, infos = env.Step(actions)
            #replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            #obs = next_obs
            
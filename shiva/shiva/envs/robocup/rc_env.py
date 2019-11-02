import numpy as np
import time
import threading
import pandas as pd
import math
import os, subprocess, time, signal
from .HFO import hfo
from .HFO.hfo import hfo as hfo_env
from .misc import zero_params
from torch.autograd import Variable
import torch

class rc_env:
    '''
        Description
            Class to run the RoboCup Environment. This class runs each agent on
            its own thread and uses Barriers to make the agents run 
            synchronously ergo take an action together at each timestep.

        Inputs
            @config     Contains various Env parameters
            @port       Required for the robocup server
    '''

    def __init__(self, config):
        self.config = config
        self.untouched = config['untouched']
        self.goalie = config['goalie']
        self.port = config['port']
        self.hfo_path = hfo.get_hfo_path()
        self.seed = np.random.randint(1000)
        self.viewer = None

        self.num_left = config['num_left']
        self.num_right = config['num_right']
        self.num_leftBot = config['num_l_bot']
        self.num_rightBot = config['num_r_bot']

        if config['action_level'] == 'low':
            #                   pow,deg   deg       deg         pow,deg    
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo_env.DASH, hfo_env.TURN, hfo_env.KICK]
            self.kick_actions = [hfo_env.KICK] # actions that require the ball to be kickable
        elif config['action_level'] == 'high':
            self.action_list = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.REORIENT, hfo_env.GO_TO_BALL, hfo_env.MOVE]
            self.kick_actions = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.PASS] # actions that require the ball to be kickable

        self.fpt = config['ep_length']

        self.act_lvl = config['action_level']
        self.feat_lvl = config['feature_level']

        if config['feature_level'] == 'low':
            #For new obs reorganization without vailds, changed hfo obs from 59 to 56
            self.left_features = 56 + 13*(self.num_left-1) + 12*self.num_right + 4 + 1 + 2 + 1
            self.right_features = 56 + 13*(self.num_right-1) + 12*self.num_left + 4 + 1 + 2 + 1
        elif config['feature_level'] == 'high':
            self.left_features = (6*self.num_left) + (3*self.num_right) + (3*self.num_rightBot) + 6
            self.right_features = (6*self.num_right) + (3*self.num_left) + (3*self.num_rightBot) + 6
        elif config['feature_level'] == 'simple':
            # 16 - land_feats + 12 - basic feats + 6 per (left/right)
            self.left_features = 28 + (6 * ((self.num_left-1) + self.num_right))
            self.right_features = 28 + (6 * ((self.num_right-1) + self.num_left))

        self.acs_dim = config['ac_dim']
        self.left_envs = None
        self.right_envs = None

        # flag that says when the episode is done
        self.d = 0
        # flag to wait for all the agents to load
        self.start = False

        # Various Barriers to keep all agents actions in sync
        self.sync_after_queue = threading.Barrier(self.num_left+self.num_right+1)
        self.sync_before_step = threading.Barrier(self.num_left+self.num_right+1)
        self.sync_at_status = threading.Barrier(self.num_left+self.num_right)
        self.sync_at_reward = threading.Barrier(self.num_left+self.num_right)

        # params for low level actions
        num_action_params = 5 # 2 for dash and kick 1 for turn and tackle

        # Left side actions, obs, rewards
        self.left_actions = np.array([2]*self.num_left)
        self.left_action_params = np.asarray([[0.0]*num_action_params for i in range(self.num_left)])
        # self.left_actions_OH = np.empty([self.num_left, 8],dtype=float)
        self.left_obs = np.empty([self.num_left,self.left_features],dtype=float)
        self.left_obs_previous = np.empty([self.num_left,self.left_features],dtype=float)
        self.left_rewards = np.zeros(self.num_left)

        # Right side actions, obs, rewards
        self.right_actions = np.array([2]*self.num_right)
        self.right_action_params = np.asarray([[0.0]*num_action_params for i in range(self.num_right)])
        # self.right_actions_OH = np.empty([self.num_right, 8],dtype=float)
        self.right_obs = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_obs_previous = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_rewards = np.zeros(self.num_right)

        self.world_status = 0
        self.left_base = 'base_left'
        self.right_base = 'base_right'
        
    def launch(self):
        '''
            Description
                1. Runs the HFO server with all of its given commands (cmd).
                2. Then runs an HFOEnv per each agent which is the bridge between
                the C++ code and python code refer to hfo.py to find this class.
                3. Both for loops create threads for each agent via the connect
                method which connects to the server and can be further described
                below.
                4. NOTE certain sleep timers are required to keep things in sync
                depending on your computers processing power you may need to
                adjust these times based off how many agents you are running.
                5. When self.start is set to True the while loop is entered
                in the connect method.
        '''

        self._start_hfo_server()
        self.left_envs = [hfo_env.HFOEnvironment() for i in range(self.num_left)]
        self.right_envs = [hfo_env.HFOEnvironment() for i in range(self.num_right)]
        
        # Create thread(s) for left side
        for i in range(self.num_left):
            print("Connecting player %i" % i , "on left %s to the server" % self.left_base)
            if i == 0:
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.left_base,
                                                self.goalie,i,self.fpt,self.act_lvl,self.left_envs,))
            else:
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.left_base,
                                                False,i,self.fpt,self.act_lvl,self.left_envs,))
            t.start()
            time.sleep(1.5)
        
        for i in range(self.num_right):
            print("Connecting player %i" % i , "on rightonent %s to the server" % self.right_base)
            if i == 0:
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.right_base,
                                                self.goalie,i,self.fpt,self.act_lvl,self.right_envs,))
            else:
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.right_base,
                                                False,i,self.fpt,self.act_lvl,self.right_envs,))
            t.start()
            time.sleep(1.5)

        print("All players connected to server")
        self.start = True

    def Observation(self,agent_id,side):
        '''
            Input
                @agent_id specifies agent ob in a given list
                @side left or right

            Returns
                Observation for the right or left depending on the side
                provided.
        '''

        if side == 'left':
            return self.left_obs[agent_id]
        elif side == 'right':
            return self.right_obs[agent_id]

    def Reward(self,agent_id,side):
        '''
            Inputs
                @agent_id specifies agent rew in a given list
                @side left or right

            Returns
                Reward for the right or left depending on the side
                provided.
        '''

        if side == 'left':
            return self.left_rewards[agent_id]
        elif side == 'right':
            return self.right_rewards[agent_id]


    def Step(self, left_actions=[], right_actions=[], left_params=[], 
            right_params=[], left_actions_OH = [], right_actions_OH = []):
        '''
            Description
                Method for the agents to take a single step in the environment.
                The actions are first queued then the Barrier waits for
                all the agents to take an action. The next Barrier syncs up all
                the agents together before taking a step and returning
                the values. Thus the while loop in connect will start another
                iteration.
            
            Inputs
                @left_actions list of left actions
                @right_actions list of right actions
                @left_params list of params corresponding to each action
                @right_params similar to left_params
                @*_OH one-hot-encoded actions
            
            Returns
                Observations
                Rewards
                Done Flag: Signals when the episode is over
                World Status: Determines if the Done Flag is set
        '''

        # for i in range(self.num_left):
        #     self.left_actions_OH[i] = misc.zero_params(left_actions_OH[i].reshape(-1))
        
        # for i in range(self.num_right):
        #     self.right_actions_OH[i] = misc.zero_params(right_actions_OH[i].reshape(-1))

        [self.Queue_action(i,self.left_base,left_actions[i],left_params) for i in range(len(left_actions))]
        [self.Queue_action(j,self.right_base,right_actions[j],right_params) for j in range(len(right_actions))]

        self.sync_after_queue.wait()
        self.sync_before_step.wait()
        
        return np.asarray(self.left_obs),self.left_rewards,np.asarray(self.right_obs),self.right_rewards, \
                self.d, self.world_status

    def Queue_action(self,agent_id,base,action,params=[]):
        '''
            Description
                Queue up the actions and params for the agents before 
                taking a step in the environment.
        '''

        if self.left_base == base:
            self.left_actions[agent_id] = action
            if self.act_lvl == 'low':
                for p in range(params.shape[1]):
                    self.left_action_params[agent_id][p] = params[agent_id][p]
        else:
            self.right_actions[agent_id] = action
            if self.act_lvl == 'low':
                for p in range(params.shape[1]):
                    self.right_action_params[agent_id][p] = params[agent_id][p]
    
    # takes param index (0-4)
    def get_valid_scaled_param(self,agentID,ac_index,base):
        '''
        Description
            
        '''

        if self.left_base == base:
            action_params = self.left_action_params
        else:
            action_params = self.right_action_params
        
        if ac_index == 0: # dash power, degree
            return (action_params[agentID][0].clip(-1,1)*100,
                    action_params[agentID][1].clip(-1,1)*180)
        elif ac_index == 1: # turn degree
            return (action_params[agentID][2].clip(-1,1)*180,)
        elif ac_index == 2: # kick power, degree
            return (((action_params[agentID][3].clip(-1,1) + 1)/2)*100,
                    action_params[agentID][4].clip(-1,1)*180)
    
    # Engineered Reward Function
    def getReward(self,s,agentID,base,ep_num):
        reward=0.0
        return reward

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self,port,feat_lvl, base, goalie, agent_ID,fpt,act_lvl,envs):
        '''
        Description
            Connect threaded agent to server. And run agents through
            environment loop ergo recieve an observation, take an action,
            recieve a new observation, world status, and reward then start
            all over again. The world status dictates if the done flag
            should change.
            
        Inputs
            feat_lvl: Feature level to use. ('high', 'low', 'simple')
            base: Which base to launch agent to. ('base_left', 'base_right)
            goalie: Play goalie. (True, False)
            agent_ID: Integer representing agent index. (0-11)
            fpt: Episode length
            act_lvl: Action level to use. ('high', 'low')
            envs: left or right agent environments

        Returns
            None, thread runs on server continually.
        '''

        if feat_lvl == 'low':
            feat_lvl = hfo_env.LOW_LEVEL_FEATURE_SET
        elif feat_lvl == 'high':
            feat_lvl = hfo_env.HIGH_LEVEL_FEATURE_SET
        elif feat_lvl == 'simple':
            feat_lvl = hfo_env.SIMPLE_LEVEL_FEATURE_SET

        config_dir = hfo.get_config_path() 
        recorder_dir = 'log/'
        envs[agent_ID].connectToServer(feat_lvl, config_dir=config_dir,
                            server_port=port, server_addr='localhost', team_name=base,
                                                play_goalie=goalie,record_dir =recorder_dir)
        
        if base == 'base_left':
            obs_prev = self.left_obs_previous
            obs = self.left_obs
            actions = self.left_actions
            rews = self.left_rewards
        else:
            obs_prev = self.right_obs_previous
            obs = self.right_obs
            actions = self.right_actions
            rews = self.right_rewards

        ep_num = 0
        while(True):
            while(self.start):
                ep_num += 1
                j = 0 # j to maximum episode length

                obs_prev[agent_ID] = envs[agent_ID].getState() # Get initial state
                obs[agent_ID] = envs[agent_ID].getState() # Get initial state

                # self.been_kicked_left = False
                # self.been_kicked_right = False
                while j < fpt:

                    self.sync_after_queue.wait()
                    
                    # take the action
                    a = actions[agent_ID]
                    if act_lvl == 'high':
                        envs[agent_ID].act(self.action_list[a]) # take the action
                    elif act_lvl == 'low':
                        # without tackle
                        # print('action:', a)
                        envs[agent_ID].act(self.action_list[a], *self.get_valid_scaled_param(agent_ID,a,base))

                    self.sync_at_status.wait()
                    
                    obs_prev[agent_ID] = obs[agent_ID]
                    self.world_status = envs[agent_ID].step() # update world                        
                    obs[agent_ID] = envs[agent_ID].getState() # update obs after all agents have acted
                    # obs[agent_ID] = actions_OH[agent_ID]

                    self.sync_at_reward.wait()

                    if self.world_status == hfo_env.IN_GAME:
                        self.d = 0
                    else:
                        self.d = 1

                    rews[agent_ID] = self.getReward(
                        envs[agent_ID].statusToString(self.world_status),agent_ID,base,ep_num) # update reward

                    j+=1
                    self.sync_before_step.wait()

                    # Break if episode done
                    if self.d == True:
                        break

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _start_hfo_server(self):
            '''
                Description
                    Runs the HFO command to pass parameters to the server. 
                    Refer to `HFO/bin/HFO` to see how these params are added.
            '''
            cmd = self.hfo_path + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                  " --log-dir %s --message-size 256"\
                  % (self.fpt, self.untouched, self.num_left,
                     self.num_right, self.num_leftBot, self.num_rightBot, self.port,
                     self.config['offense_ball'], self.seed, self.config['ball_x_min'], self.config['ball_x_max'],
                     self.config['ball_y_min'], self.config['ball_y_max'], self.config['log'])
            #Adds the binaries when offense and defense npcs are in play, must be changed to add agent vs binary npc
            if self.num_leftBot > 0:   cmd += " --offense-left %s" \
                % (self.config['left_bin'])
            if self.num_rightBot > 0:   cmd += " --defense-left %s" \
                % (self.config['right_bin'])
            if not self.config['sync_mode']:      cmd += " --no-sync"
            if self.config['fullstate']:          cmd += " --fullstate"
            if self.config['determ']:      cmd += " --deterministic"
            if self.config['verbose']:            cmd += " --verbose"
            if not self.config['rcss_log']:  cmd += " --no-logging"
            if self.config['hfo_log']:       cmd += " --hfo-logging"
            if self.config['record_lib']:             cmd += " --record"
            if self.config['record_serv']:      cmd += " --log-gen-pt"
            if self.config['init_env']:
                cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                        " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                        " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                        % (self.config['agents_x_min'], self.config['agents_x_max'], self.config['agents_y_min'], self.config['agents_y_max'],
                            self.config['change_every_x'], self.config['change_agents_x'], self.config['change_agents_y'],
                            self.config['change_ball_x'], self.config['change_ball_y'])

            print('Starting server with command: %s' % cmd)
            self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
            time.sleep(3) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        '''
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        '''
        
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)
        cmd = hfo.get_viewer_path() +\
              " --connect --port %d" % (self.port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)


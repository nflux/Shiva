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

possession_side = 'N'

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

        self.rew_anneal_ep = config['reward_anneal']

        self.num_left = config['num_left']
        self.num_right = config['num_right']
        self.num_leftBot = config['num_l_bot']
        self.num_rightBot = config['num_r_bot']

        if config['action_level'] == 'low':
            #                   pow,deg   deg       deg         pow,deg    
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo_env.DASH, hfo_env.TURN, hfo_env.KICK]
            self.kick_actions = [hfo_env.KICK] # actions that require the ball to be kickable
            self.acs_param_dim = 5
        elif config['action_level'] == 'high':
            self.action_list = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.REORIENT, hfo_env.GO_TO_BALL, hfo_env.MOVE]
            self.kick_actions = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.PASS] # actions that require the ball to be kickable
            # self.acs_param_dim = ????

        self.acs_dim = len(self.action_list)

        self.fpt = config['ep_length']

        self.act_lvl = config['action_level']
        self.feat_lvl = config['feature_level']

        self.set_observation_indexes(config['feature_level'])

        if config['feature_level'] == 'low':
            #For new obs reorganization without vailds, changed hfo obs from 59 to 56
            # self.left_features = 56 + 13*(self.num_left-1) + 12*self.num_right + 4 + 1 + 2 + 1
            # self.right_features = 56 + 13*(self.num_right-1) + 12*self.num_left + 4 + 1 + 2 + 1
            self.left_features = 16
            self.right_features = 16
        elif config['feature_level'] == 'high':
            self.left_features = (6*self.num_left) + (3*self.num_right) + (3*self.num_rightBot) + 6
            self.right_features = (6*self.num_right) + (3*self.num_left) + (3*self.num_rightBot) + 6
        elif config['feature_level'] == 'simple':
            # 16 - land_feats + 12 - basic feats + 6 per (left/right)
            # Feature indexes by name
            self.left_features = 28 + (6 * ((self.num_left-1) + self.num_right))
            self.right_features = 28 + (6 * ((self.num_right-1) + self.num_left))

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
        self.left_kickable = [0] * self.num_left
        self.left_agent_possession = ['N'] * self.num_left
        self.left_passer = [0]*self.num_left
        self.left_lost_possession = [0]*self.num_left

        # Right side actions, obs, rewards
        self.right_actions = np.array([2]*self.num_right)
        self.right_action_params = np.asarray([[0.0]*num_action_params for i in range(self.num_right)])
        # self.right_actions_OH = np.empty([self.num_right, 8],dtype=float)
        self.right_obs = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_obs_previous = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_rewards = np.zeros(self.num_right)
        self.right_kickable = [0] * self.num_right
        self.right_agent_possession = ['N'] * self.num_right
        self.right_passer = [0]*self.num_right
        self.right_lost_possession = [0]*self.num_right

        self.world_status = 0
        self.left_base = 'base_left'
        self.right_base = 'base_right'


        self.left_agent_possesion = ['N'] * self.num_left

    def set_observation_indexes(self, feature_level):

        if feature_level == 'low':

            self.ball_x = 0
            self.ball_y = 1
            self.goal_x = 2
            self.goal_y = 3
            self.x = 4
            self.y = 5
            self.stamina = 6
            self.kickable = 7

            # '''
            #     Agent Values
            
            # '''
            # self.agent_angle_vel_y_rad = 1
            # self.agent_angle_vel_x_rad = 2
            # self.agent_vel_mag = 3
            # self.agent_angle_global_x_rad = 4
            # self.agent_angle_global_y_rad = 5
            # self.agent_stamina = 6
            # self.agent_frozen = 7
            # self.agent_collision_with_ball = 8
            # self.agent_collision_with_player = 9
            # self.agent_collision_with_post = 10
            # self.agent_kickable = 11

            # '''
            #     Goal Values

            # '''
            # # center of goal
            # self.goal_opp_cent_y = 12
            # self.goal_opp_cent_x = 13
            # self.goal_opp_cent_proximity = 14

            # # top of goal
            # self.goal_opp_top_y = 15
            # self.goal_opp_top_x = 16
            # self.goal_opp_top_proximity = 17

            # # bottom of goal
            # self.goal_opp_bot_y = 18
            # self.goal_opp_bot_x = 19
            # self.goal_opp_bot_proximity = 20

            # '''
            #     Field Values
            
            # '''
            # # center of field
            # self.field_center_y_rad = 30
            # self.field_center_x_rad = 31
            # self.field_center_proximity = 32

            # # top left field
            # self.field_top_left_y_rad = 33
            # self.field_top_left_x_rad = 34
            # self.field_top_left_proximity = 35

            # # top right field
            # self.field_top_right_y_rad = 36
            # self.field_top_right_x_rad = 37
            # self.field_top_right_proximity = 38

            # # bottom right field
            # self.field_bot_right_y_rad = 39
            # self.field_bot_right_x_rad = 40
            # self.field_bot_right_proximity = 41

            # # bottom left field
            # self.field_bot_left_y_rad = 42
            # self.field_bot_left_x_rad = 43
            # self.field_bot_left_proximity = 44

            # # proximities to sides
            # self.field_left_proximity = 45
            # self.field_right_proximity = 46
            # self.field_top_proximity = 47
            # self.field_bottom_proximity = 48

            # '''
            #     Ball Values

            # '''
            # self.ball_angle_y_rad = 49
            # self.ball_angle_x_rad = 50
            # self.ball_proximity = 51  
            # self.ball_vel_mag = 52
            # self.ball_vel_y_rad = 53
            # self.ball_vel_x_rad = 54
            # self.ball_x = 55
            # self.ball_y = 56

        elif feature_level == 'simple':
            self.stamina = 26
            self.ball_x = 16
            self.ball_y = 17
            self.ball_x_vel = 18
            self.ball_y_vel = 19
            self.x = 20
            self.y = 21
            self.x_vel = 22
            self.y_vel = 23
            self.opp_goal_top_x = 12
            self.opp_goal_top_y = 13
            self.opp_goal_bot_x = 14
            self.opp_goal_bot_y = 15

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

    def getImitObsMsg(self):
        obsMsg = ""
        obsMsg += str(self.left_envs[0].getBallX()) + " "
        obsMsg += str(self.left_envs[0].getBallY()) + " "
        obsMsg += str(self.left_envs[0].getBallVelX()) + " "
        obsMsg += str(self.left_envs[0].getBallVelY()) + " "

        for env in self.left_envs:
            obsMsg += str(env.side()) + " "
            obsMsg += str(env.getUnum()) + " "
            obsMsg += str(env.getSelfX()) + " "
            obsMsg += str(env.getSelfY()) + " "
            obsMsg += str(env.getSelfAng()) + " "
            obsMsg += str(env.getSelfVelX()) + " "
            obsMsg += str(env.getSelfVelY()) + " "
        
        for env in self.right_envs:
            obsMsg += str(env.side()) + " "
            obsMsg += str(env.getUnum()) + " "
            obsMsg += str(env.getSelfX()) + " "
            obsMsg += str(env.getSelfY()) + " "
            obsMsg += str(env.getSelfAng()) + " "
            obsMsg += str(env.getSelfVelX()) + " "
            obsMsg += str(env.getSelfVelY()) + " "

        return str(obsMsg).encode("utf-8")

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
        
        return self.left_obs, self.left_rewards, self.right_obs, self.right_rewards, self.d, self.world_status

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
                    # print(a)
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
                        envs[agent_ID].statusToString(self.world_status),
                        agent_ID,
                        base,
                        ep_num
                    ) # update reward

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
                  " --log-dir %s --seed %i --message-size 256"\
                  % (self.fpt, self.untouched, self.num_left,
                     self.num_right, self.num_leftBot, self.num_rightBot, self.port,
                     self.config['offense_ball'], self.seed, self.config['ball_x_min'], self.config['ball_x_max'],
                     self.config['ball_y_min'], self.config['ball_y_max'], self.config['log'], self.config['seed'])
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
            if self.config['run_imit']:         cmd += " --run-bots"
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

    def getReward(self, s, agentID, base, ep_num):
        '''
            Reward Engineering - Needs work!

            Input
                s           world status message
                agentId     
                base        left_base or right_base
                ep_num      episode number
            
        '''
        reward=0.0
        team_reward = 0.0
        goal_points = 20.0
        #---------------------------
        global possession_side
        if self.d:
            if self.left_base == base:
            # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----

                if s=='Goal_By_Left' and self.left_agent_possesion[agentID] == 'L':
                    reward+= goal_points
                elif s=='Goal_By_Left':
                    reward+= goal_points # teammates get 10% of points

                # doesn't seem that this if statement is working
                elif s=='Goal_By_Right':
                    reward+=-goal_points

                elif s=='OutOfBounds' and self.left_agent_possesion[agentID] == 'L':
                    reward+=-0.5
                elif s=='CapturedByLeftGoalie':
                    reward+=goal_points/5.0
                elif s=='CapturedByRightGoalie':
                    reward+= 0 #-goal_points/4.0
                possession_side = 'N' # at the end of each episode we set this to none
                self.left_agent_possesion = ['N'] * self.num_left
                return reward
            else:
                if s=='Goal_By_Right' and self.right_agent_possesion[agentID] == 'R':
                    reward+=goal_points
                elif s=='Goal_By_Right':
                    reward+=goal_points
                elif s=='Goal_By_Left':
                    reward+=-goal_points
                elif s=='OutOfBounds' and self.right_agent_possesion[agentID] == 'R':
                    reward+=-0.5
                elif s=='CapturedByRightGoalie':
                    reward+=goal_points/5.0
                elif s=='CapturedByLeftGoalie':
                    reward+= 0 #-goal_points/4.0

                possession_side = 'N'
                self.right_agent_possesion = ['N'] * self.num_right
                return reward
        
        if self.left_base == base:
            team_actions = self.left_actions
            team_obs = self.left_obs
            team_obs_previous = self.left_obs_previous
            opp_obs = self.right_obs
            opp_obs_previous = self.right_obs_previous
            num_ag = self.num_left
            env = self.left_envs[agentID]
            kickable = self.left_kickable[agentID]
            self.left_kickable[agentID] = self.get_kickable_status(agentID,env)
        else:
            team_actions = self.right_actions
            team_obs = self.right_obs
            team_obs_previous = self.right_obs_previous
            opp_obs = self.left_obs
            opp_obs_previous = self.left_obs_previous
            num_ag = self.num_right
            env = self.right_envs[agentID]
            kickable = self.right_kickable[agentID]
            self.right_kickable[agentID] = self.get_kickable_status(agentID,env)# update kickable status (it refers to previous timestep, e.g., it WAS kickable )

        # Low stamina - seems that needs to be implemented (Ezequiel)

        if team_obs[agentID][self.stamina] < 0.0 : # LOW STAMINA
            reward -= 1
            team_reward -= 1
            # print ('low stamina')
            pass
        
        # print('sin:', team_obs[agentID][49], '\ncos:', team_obs[agentID][50], '\nprox:', team_obs[agentID][51])

        ############ Kicked Ball #################

        # print(kickable)
        # print(self.action_list[team_actions[agentID]] in self.kick_actions)
        # print(self.num_right > 0)
        self.right_agent_possesion = 'N'


        if self.action_list[team_actions[agentID]] in self.kick_actions and not kickable:
            reward -= 1

        
        if self.action_list[team_actions[agentID]] in self.kick_actions and kickable:    
            if True:        
            # if self.num_right > 0:
                if (np.array(self.left_agent_possesion) == 'N').all() and (np.array(self.right_agent_possesion) == 'N').all():
                    # print("First Kick")
                    reward += 1.5
                    team_reward += 1.5
                # set initial ball position after kick
                    if self.left_base == base:
                        self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
                        self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
                    else:
                        self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
                        self.BR_ball_pos_y = team_obs[agentID][self.ball_y]
                        

            # track ball delta in between kicks
            if self.left_base == base:
                self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
                self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
            else:
                self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
                self.BR_ball_pos_y = team_obs[agentID][self.ball_y]

            new_x = team_obs[agentID][self.ball_x]
            new_y = team_obs[agentID][self.ball_y]
            
            if self.left_base == base:
                ball_delta = math.sqrt((self.BL_ball_pos_x-new_x)**2+ (self.BL_ball_pos_y-new_y)**2)
                self.BL_ball_pos_x = new_x
                self.BL_ball_pos_y = new_y
            else:
                ball_delta = math.sqrt((self.BR_ball_pos_x-new_x)**2+ (self.BR_ball_pos_y-new_y)**2)
                self.BR_ball_pos_x = new_x
                self.BR_ball_pos_y = new_y
            
            self.pass_reward = ball_delta * 5.0

        #     ######## Pass Receiver Reward #########
            if self.left_base == base:
                if (np.array(self.left_agent_possesion) == 'L').any():
                    prev_poss = (np.array(self.left_agent_possesion) == 'L').argmax()
                    if not self.left_agent_possesion[agentID] == 'L':
                        self.left_passer[prev_poss] += 1 # sets passer flag to whoever passed
                        # Passer reward is added in step function after all agents have been checked
                       
                        # reward += self.pass_reward
                        # team_reward += self.pass_reward
                        #print("received a pass worth:",self.pass_reward)
        #               #print('team pass reward received ')
        #         #Remove this check when npc ball posession can be measured
                if self.num_right > 0:
                    if (np.array(self.right_agent_possesion) == 'R').any():
                        enemy_possessor = (np.array(self.right_agent_possesion) == 'R').argmax()
                        self.right_lost_possession[enemy_possessor] -= 1.0
                        self.left_lost_possession[agentID] += 1.0
                        # print('BR lost possession')
                        self.pass_reward = 0

        #         ###### Change Possession Reward #######
                self.left_agent_possesion = ['N'] * self.num_left
                self.right_agent_possesion = ['N'] * self.num_right
                self.left_agent_possesion[agentID] = 'L'
                if possession_side != 'L':
                    possession_side = 'L'    
                    #reward+=1
                    #team_reward+=1
            else:
                # self.opp_possession_counter[agentID] += 1
                if (np.array(self.right_agent_possesion) == 'R').any():
                    prev_poss = (np.array(self.right_agent_possesion) == 'R').argmax()
                    if not self.right_agent_possesion[agentID] == 'R':
                        self.right_passer[prev_poss] += 1 # sets passer flag to whoever passed
                        # reward += self.pass_reward
                        # team_reward += self.pass_reward
                        # print('opp pass reward received ')

                if (np.array(self.left_agent_possesion) == 'L').any():
                    enemy_possessor = (np.array(self.left_agent_possesion) == 'L').argmax()
                    self.left_lost_possession[enemy_possessor] -= 1.0
                    self.right_lost_possession[agentID] += 1.0
                    self.pass_reward = 0
      #             # print('BL lost possession ')

                self.left_agent_possesion = ['N'] * self.num_left
                self.right_agent_possesion = ['N'] * self.num_right
                self.right_agent_possesion[agentID] = 'R'
                if possession_side != 'R':
                    possession_side = 'R'
                    #reward+=1
                    #team_reward+=1

        ####################### reduce distance to ball - using delta  ##################
        # all agents rewarded for closer to ball
        # dist_cur = self.distance_to_ball(team_obs[agentID])
        # dist_prev = self.distance_to_ball(team_obs_previous[agentID])
        # d = (0.5)*(dist_prev - dist_cur) # if cur > prev --> +   
        # if delta > 0:
        #     reward  += delta
        #     team_reward += delta
            
        ####################### Rewards the closest player to ball for advancing toward ball ############
        distance_cur, closest_agent = self.closest_player_to_ball(team_obs, num_ag)
        distance_prev, _ = self.closest_player_to_ball(team_obs_previous, num_ag)
        if agentID == closest_agent:
            delta = (distance_prev - distance_cur)*1.0
            #if delta > 0:    
            if True:
                team_reward += delta * 10 
                reward+= delta * 10
                # print(distance_cur, delta)
                pass
            
        ##################################################################################
            
        ####################### reduce ball distance to goal ##################
        # base left kicks
        r = self.ball_distance_to_goal(team_obs[agentID]) 
        r_prev = self.ball_distance_to_goal(team_obs_previous[agentID]) 
        if ((self.left_base == base) and possession_side =='L'):
            team_possessor = (np.array(self.left_agent_possesion) == 'L').argmax()
            if agentID == team_possessor:
                delta = (2*self.num_left)*(r_prev - r)
                if True:
                #if delta > 0:
                    reward += delta
                    team_reward += delta
                    pass

        # base right kicks
        elif self.right_base == base and possession_side == 'R':
            team_possessor = (np.array(self.right_agent_possesion) == 'R').argmax()
            if agentID == team_possessor:
                delta = (2*self.num_left)*(r_prev - r)
                if True:
                #if delta > 0:
                    # reward += delta
                    # team_reward += delta
                    pass
        # non-possessor reward for ball delta toward goal
        else:
            delta = (0*self.num_left)*(r_prev - r)
            if True:
            #if delta > 0:
                # reward += delta
                # team_reward += delta  
                pass  
        '''
            Reward agent for maximizing it's proximity to the ball
        '''
        # reward += team_obs[agentID][self.ball_proximity]

        # print(team_obs[agentID][self.ball_x])
        # print(team_obs[agentID][self.ball_y])
        return reward
        # rew_percent = 1.0*max(0,(self.rew_anneal_ep - ep_num))/self.rew_anneal_ep
        # return ((1.0 - rew_percent)*team_reward) + (reward * rew_percent)

    '''

        Below are function utilities for the Reward Engineering

    '''

    def get_kickable_status(self,agentID,env):
        ball_kickable = False
        ball_kickable = env.isKickable()
        #print("no implementation")
        return ball_kickable

    def closest_player_to_ball(self, team_obs, num_agents):
        '''
        teams receive reward based on the distance of their closest agent to the ball
        '''
        closest_player_index = 0
        ball_distance = self.distance_to_ball(team_obs[0])
        for i in range(1, num_agents):
            temp_distance = self.distance_to_ball(team_obs[i])
            if temp_distance < ball_distance:
                closest_player_index = i
                ball_distance = temp_distance
        return ball_distance, closest_player_index

    def distance_to_ball(self, obs):
        relative_x = obs[self.x]-obs[self.ball_x]
        relative_y = obs[self.y]-obs[self.ball_y]
        ball_distance = math.sqrt(relative_x**2+relative_y**2)
        
        return ball_distance

    def ball_distance_to_goal(self,obs):
        goal_center_x = 1.0
        goal_center_y = 0.0
        relative_x = obs[self.ball_x] - goal_center_x
        relative_y = obs[self.ball_y] - goal_center_y
        ball_distance_to_goal = math.sqrt(relative_x**2 + relative_y**2)
        return ball_distance_to_goal

    def prox_2_dist(self, prox):
        return (prox+.8)/1.8
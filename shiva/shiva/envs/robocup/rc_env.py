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

        if 'MetaLearner' in config:
            {setattr(self, k, v) for k,v in config['Environment'].items()}
        else:
            {setattr(self, k, v) for k,v in config.items()}

        self.hfo_path = hfo.get_hfo_path()
        # self.seed = np.random.randint(1000)
        self.viewer = None

        self.left_action_option = None
        self.right_action_option = None

        if self.action_level == 'low':
            #                   pow,deg   deg       deg         pow,deg
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo_env.DASH, hfo_env.TURN, hfo_env.KICK]
            self.kick_actions = [hfo_env.KICK] # actions that require the ball to be kickable
            self.acs_dim = len(self.action_list)
            self.acs_param_dim = 5 # 2 for dash and kick 1 for turn and tackle
            self.left_action_option = np.asarray([[0.0]*self.acs_param_dim for i in range(self.num_left)], dtype=np.float64)
            # self.left_actions_OH = np.empty([self.num_left, 8],dtype=float)
            self.right_action_option = np.asarray([[0.0]*self.acs_param_dim for i in range(self.num_right)], dtype=np.float64)
            # self.right_actions_OH = np.empty([self.num_right, 8],dtype=float)
        elif self.action_level == 'discretized':

            self.action_list = [hfo_env.DASH , hfo_env.TURN , hfo_env.KICK]
            dash_power_discretization = np.linspace(-100,100,21, dtype=np.float64).tolist()
            dash_degree_discretization = np.linspace(-180,180,9, dtype=np.float64).tolist()
            power_discretization = np.linspace(0,100,21, dtype=np.float64).tolist()
            degree_discretization = np.linspace(-180,180,17, dtype=np.float64).tolist()

            self.dash_pow_step = dash_power_discretization[1] - dash_power_discretization[0]
            self.dash_degree_step = dash_degree_discretization[1] - dash_degree_discretization[0]
            self.pow_step = power_discretization[1]-power_discretization[0]
            self.degree_step = degree_discretization[1]-degree_discretization[0]

            # Reference Tables
            # self.DASH_TABLE = []
            # self.KICK_TABLE = []
            # self.TURN_TABLE = self.degree_discretization
            self.ACTION_DICT = {}
            self.REVERSE_ACTION_DICT = {}
            dis_ctr = 0
            rev_ctr = 0
            turn_dict = kick_dict = {}
            for dash_power in dash_power_discretization:
                for dash_degree in dash_degree_discretization:
                    self.ACTION_DICT[dis_ctr] = (dash_power, dash_degree)
                    dis_ctr += 1

            self.dash_idx = dis_ctr

            self.REVERSE_ACTION_DICT[rev_ctr] = dict(zip(self.ACTION_DICT.values(), self.ACTION_DICT.keys()))
            rev_ctr += 1
            for turn_degree in degree_discretization:
                turn_dict[dis_ctr] = (turn_degree,)
                self.ACTION_DICT[dis_ctr] = (turn_degree,)
                dis_ctr += 1

            self.turn_idx = dis_ctr

            self.REVERSE_ACTION_DICT[rev_ctr] = dict(zip(turn_dict.values(), turn_dict.keys()))
            rev_ctr += 1
            for kick_power in power_discretization:
                for kick_degree in degree_discretization:
                    kick_dict[dis_ctr] = (kick_power, kick_degree)
                    self.ACTION_DICT[dis_ctr] = (kick_power, kick_degree)
                    dis_ctr += 1

            self.REVERSE_ACTION_DICT[rev_ctr] = dict(zip(kick_dict.values(), kick_dict.keys()))
            # for a in range(len(self.action_list)):
            #     if a == 0:
            #         self.ACTION_DICT[a] = dash_dict
            #     elif a == 1:
            #         self.ACTION_DICT[a] = turn_dict
            #     else:
            #         self.ACTION_DICT[a] = kick_dict

            # for dash_power in self.power_discretization:
            #     for dash_degree in self.degree_discretization:
            #         self.DASH_TABLE.append((dash_power,dash_degree))
            #         self.KICK_TABLE.append((dash_power,dash_degree))


            self.left_action_option = [0]*self.num_left
            self.right_action_option = [0]*self.num_right

            # self.ACTION_MATRIX = self.DASH_TABLE + self.TURN_TABLE + self.KICK_TABLE
            # self.kick_actions = self.KICK_TABLE

            # self.acs_dim = len(self.DASH_TABLE) + len(self.TURN_TABLE) + len(self.KICK_TABLE)
            self.acs_dim =  dis_ctr
            self.acs_param_dim = 0

        elif self.action_level == 'high':
            self.action_list = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.REORIENT, hfo_env.GO_TO_BALL, hfo_env.MOVE]
            self.kick_actions = [hfo_env.DRIBBLE, hfo_env.SHOOT, hfo_env.PASS] # actions that require the ball to be kickable

        self.num_actions = len(self.action_list)
        self.set_observation_indexes()

        if self.feature_level == 'low':
            #For new obs reorganization without vailds, changed hfo obs from 59 to 56
            # self.left_features = 56 + 13*(self.num_left-1) + 12*self.num_right + 4 + 1 + 2 + 1
            # self.right_features = 56 + 13*(self.num_right-1) + 12*self.num_left + 4 + 1 + 2 + 1
            self.left_features = 16
            self.right_features = 16
        elif self.feature_level == 'high':
            self.left_features = (6*self.num_left) + (3*self.num_right) + (3*self.num_r_bot) + 6
            self.right_features = (6*self.num_right) + (3*self.num_left) + (3*self.num_r_bot) + 6
        elif self.feature_level == 'simple':
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

        self.close = False

        # Various Barriers to keep all agents actions in sync
        self.sync_after_queue = threading.Barrier(self.num_left+self.num_right+1)
        self.sync_before_step = threading.Barrier(self.num_left+self.num_right+1)
        self.sync_at_status = threading.Barrier(self.num_left+self.num_right)
        self.sync_at_reward = threading.Barrier(self.num_left+self.num_right)

        # Left side actions, obs, rewards
        self.left_actions = np.array([0]*self.num_left, dtype=int)
        self.left_obs = np.zeros([self.num_left,self.left_features],dtype=np.float64)
        self.left_obs_previous = np.zeros([self.num_left,self.left_features],dtype=np.float64)
        self.left_rewards = np.zeros(self.num_left, dtype=np.float64)
        self.left_kickable = [0] * self.num_left
        self.left_agent_possession = ['N'] * self.num_left
        self.left_passer = [0]*self.num_left
        self.left_lost_possession = [0]*self.num_left

        # Right side actions, obs, rewards
        self.right_actions = np.array([0]*self.num_right, dtype=int)
        self.right_obs = np.zeros([self.num_right,self.right_features],dtype=np.float64)
        self.right_obs_previous = np.zeros([self.num_right,self.right_features],dtype=np.float64)
        self.right_rewards = np.zeros(self.num_right, dtype=np.float64)
        self.right_kickable = [0] * self.num_right
        self.right_agent_possession = ['N'] * self.num_right
        self.right_passer = [0]*self.num_right
        self.right_lost_possession = [0]*self.num_right

        self.world_status = 0
        self.left_base = 'base_left'
        self.right_base = 'base_right'

        self.left_agent_possesion = ['N'] * self.num_left
        self.right_agent_possesion = ['N'] * self.num_right

    def set_observation_indexes(self):

        if self.feature_level == 'low':

            self.ball_x = 0
            self.ball_y = 1
            self.goal_x = 2
            self.goal_y = 3
            self.x = 4
            self.y = 5
            self.stamina = 6
            self.kickable = 7

        elif self.feature_level == 'simple':
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
                t = threading.Thread(target=self.connect, args=(self.port,self.feature_level, self.left_base,
                                                self.left_goalie,i,self.ep_length,self.action_level,self.left_envs,))
            else:
                t = threading.Thread(target=self.connect, args=(self.port,self.feature_level, self.left_base,
                                                False,i,self.ep_length,self.action_level,self.left_envs,))
            t.start()
            time.sleep(3)

        for i in range(self.num_right):
            print("Connecting player %i" % i , "on rightonent %s to the server" % self.right_base)
            if i == 0:
                t = threading.Thread(target=self.connect, args=(self.port,self.feature_level, self.right_base,
                                                self.right_goalie,i,self.ep_length,self.action_level,self.right_envs,))
            else:
                t = threading.Thread(target=self.connect, args=(self.port,self.feature_level, self.right_base,
                                                False,i,self.ep_length,self.action_level,self.right_envs,))
            t.start()
            time.sleep(3)

        print("All players connected to server")
        self.start = True

    def start_env(self):
        return self.start

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
            obsMsg += str(env.getStamina()) + " "

        # for env in self.right_envs:
        #     obsMsg += str(env.side()) + " "
        #     obsMsg += str(env.getUnum()) + " "
        #     obsMsg += str(env.getSelfX()) + " "
        #     obsMsg += str(env.getSelfY()) + " "
        #     obsMsg += str(env.getSelfAng()) + " "
        #     obsMsg += str(env.getSelfVelX()) + " "
        #     obsMsg += str(env.getSelfVelY()) + " "
        #     obsMsg += str(env.getStamina()) + " "

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


    def Step(self, left_actions=[], right_actions=[], left_options=[],
            right_options=[], left_actions_OH = [], right_actions_OH = []):
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

        [self.Queue_action(i,self.left_base,left_actions[i],left_options) for i in range(len(left_actions))]
        [self.Queue_action(j,self.right_base,right_actions[j],right_options) for j in range(len(right_actions))]

        self.sync_after_queue.wait()
        self.sync_before_step.wait()

        return self.left_obs, self.left_rewards, self.right_obs, self.right_rewards, self.d, self.world_status

    def Queue_action(self,agent_id,base,action,options):
        '''
            Description
                Queue up the actions and params for the agents before
                taking a step in the environment.
        '''

        if self.left_base == base:
            self.left_actions[agent_id] = action
            if self.action_level == 'low':
                for p in range(options.shape[1]):
                    self.left_action_option[agent_id][p] = options[agent_id][p]
            # i was thinking that maybe I could choose the action here
            elif self.action_level == 'discretized':
                self.left_action_option[agent_id] = options[agent_id]
        else:
            self.right_actions[agent_id] = action
            if self.action_level == 'low':
                for p in range(options.shape[1]):
                    self.right_action_option[agent_id][p] = options[agent_id][p]

    def descritize_action(self, action):
        '''
        Descritize a parameterized action
        '''
        act_choice = np.argmax(action[:self.num_actions])
        params = action[self.num_actions:]

        if act_choice == 0: # Dash
            power = params[0].clip(-1,1)*100
            degree = params[1].clip(-1,1)*180
            return self.REVERSE_ACTION_DICT[act_choice][((self.dash_pow_step*np.round(power/self.dash_pow_step)), (self.dash_degree_step*np.round(degree/self.dash_degree_step)))]
        elif act_choice == 1: # Turn
            degree = params[2].clip(-1,1)*180
            return self.REVERSE_ACTION_DICT[act_choice][((self.degree_step*np.round(degree/self.degree_step)),)]
        else: # Kick
            power = ((params[3].clip(-1,1) + 1)/2)*100
            degree = params[4].clip(-1,1)*180
            return self.REVERSE_ACTION_DICT[act_choice][((self.pow_step*np.round(power/self.pow_step)), (self.degree_step*np.round(degree/self.degree_step)))]

    def get_valid_discrete_value(self, agentID, base):
        if self.left_base == base:
            discrete_action = self.left_action_option[agentID]
        else:
            discrete_action = self.right_action_option[agentID]

        return self.ACTION_DICT[discrete_action]

    # takes param index (0-4)
    def get_valid_scaled_param(self,agentID,ac_index,base):
        '''
            Description

        '''

        if self.left_base == base:
            action_params = self.left_action_option
        else:
            action_params = self.right_action_option

        if ac_index == 0: # dash power, degree

            dash_power =  action_params[agentID][0].clip(-1,1)*100
            dash_degree = action_params[agentID][1].clip(-1,1)*180
            return (dash_power, dash_degree)

        elif ac_index == 1: # turn degree

            turn_degree = action_params[agentID][2].clip(-1,1)*180
            return (turn_degree,)

        elif ac_index == 2: # kick power, degree

            kick_power = ((action_params[agentID][3].clip(-1,1) + 1)/2)*100
            kick_degree = action_params[agentID][4].clip(-1,1)*180
            return (kick_power, kick_degree)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def connect(self,port,feat_lvl, base, goalie, agent_ID,ep_length,act_lvl,envs):
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
            ep_length: Episode length
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
        envs[agent_ID].connectToServer(feat_lvl, config_dir=config_dir,
                            server_port=port, server_addr='localhost', team_name=base,
                                                play_goalie=goalie,record_dir =self.rc_log+'/')

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
                while j < ep_length:

                    self.sync_after_queue.wait()

                    # take the action
                    a = actions[agent_ID]

                    if act_lvl == 'high':
                        envs[agent_ID].act(self.action_list[a]) # take the action
                    elif act_lvl == 'low':
                        # without tackle
                        envs[agent_ID].act(self.action_list[a], *self.get_valid_scaled_param(agent_ID,a,base))
                    elif act_lvl == 'discretized':
                        envs[agent_ID].act(self.action_list[a], *self.get_valid_discrete_value(agent_ID,base))

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
            if self.close:
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
                " --port %i --offense-on-ball %i --ball-x-min %f"\
                " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                " --logs-dir %s --seed %i --message-size 256 --tackle-cycles 1 --no-offside --offside-area-size 0"\
                % (self.ep_length, self.untouched, self.num_left,
                    self.num_right, self.num_l_bot, self.num_r_bot, self.port,
                    self.offense_ball, self.ball_x_min, self.ball_x_max,
                    self.ball_y_min, self.ball_y_max, self.rc_log, self.seed)
        #Adds the binaries when offense and defense npcs are in play, must be changed to add agent vs binary npc
        if self.num_l_bot > 0:   cmd += " --offense-team %s" \
            % (self.left_bin)
        if self.num_r_bot > 0:   cmd += " --defense-team %s" \
            % (self.right_bin)
        if not self.sync_mode:      cmd += " --no-sync"
        if self.fullstate:          cmd += " --fullstate"
        if self.determ:      cmd += " --deterministic"
        if self.verbose:            cmd += " --verbose"
        if not self.rcss_log:  cmd += " --no-logging"
        if self.hfo_log:       cmd += " --hfo-logging"
        if self.record_lib:             cmd += " --record"
        if self.record_serv:      cmd += " --logs-gen-pt"
        if self.init_env:
            cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                    " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                    " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                    % (self.agents_x_min, self.agents_x_max, self.agents_y_min, self.agents_y_max,
                        self.change_every_x, self.change_agents_x, self.change_agents_y,
                        self.change_ball_x, self.change_ball_y)

        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(6) # Wait for server to startup before connecting a player

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

    def checkGoal(self):
        return self.left_envs[0].statusToString(self.world_status) == 'Goal_By_Left'

    def getReward(self, s, agentID, base, ep_num):
        '''
            Reward Engineering - Needs work!

            Input
                s           world status message
                agentId
                base        left_base or right_base
                ep_num      episode number

        '''
        return 0
    #     reward=0.0
    #     team_reward = 0.0
    #     goal_points = 10.0
    #     #---------------------------
    #     global possession_side
    #     if self.d:
    #         if self.left_base == base:
    #         # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----

    #             if s=='Goal_By_Left' and self.left_agent_possesion[agentID] == 'L':
    #                 reward+= goal_points
    #             elif s=='Goal_By_Left':
    #                 reward+= goal_points # teammates get 10% of pointsssss
    #                 # print("GOAL!")
    #             elif s=='Goal_By_Right':
    #                 reward+=-goal_points
    #             elif s=='OutOfBounds' and self.left_agent_possesion[agentID] == 'L':
    #                 reward+=-0.5
    #             elif s=='CapturedByLeftGoalie':
    #                 reward+=goal_points/5.0
    #             elif s=='CapturedByRightGoalie':
    #                 reward+= 0 #-goal_points/4.0
    #             possession_side = 'N' # at the end of each episode we set this to none
    #             self.left_agent_possesion = ['N'] * self.num_left
    #             return reward
    #         else:
    #             if s=='Goal_By_Right' and self.right_agent_possesion[agentID] == 'R':
    #                 reward+=goal_points
    #             elif s=='Goal_By_Right':
    #                 reward+=goal_points
    #             elif s=='Goal_By_Left':
    #                 reward+=-goal_points
    #             elif s=='OutOfBounds' and self.right_agent_possesion[agentID] == 'R':
    #                 reward+=-0.5
    #             elif s=='CapturedByRightGoalie':
    #                 reward+=goal_points/5.0
    #             elif s=='CapturedByLeftGoalie':
    #                 reward+= 0 #-goal_points/4.0

    #             possession_side = 'N'
    #             return reward

    #     if self.left_base == base:
    #         team_actions = self.left_actions
    #         team_obs = self.left_obs
    #         team_obs_previous = self.left_obs_previous
    #         opp_obs = self.right_obs
    #         opp_obs_previous = self.right_obs_previous
    #         num_ag = self.num_left
    #         env = self.left_envs[agentID]
    #         kickable = self.left_kickable[agentID]
    #         self.left_kickable[agentID] = self.get_kickable_status(agentID,env)
    #     else:
    #         team_actions = self.right_actions
    #         team_obs = self.right_obs
    #         team_obs_previous = self.right_obs_previous
    #         opp_obs = self.left_obs
    #         opp_obs_previous = self.left_obs_previous
    #         num_ag = self.num_right
    #         env = self.right_envs[agentID]
    #         kickable = self.right_kickable[agentID]
    #         self.right_kickable[agentID] = self.get_kickable_status(agentID,env)# update kickable status (it refers to previous timestep, e.g., it WAS kickable )

    #     # so this appears to be working, maybe just because the agent doesn't really have to run to it
    #     # will verify after I fix HPI
    #     if team_obs[agentID][self.stamina] < 0.0 : # LOW STAMINA
    #         reward -= 1
    #         team_reward -= 1
    #         # print("agent is getting penalized for having low stamina")

    #     ############ Kicked Ball #################

    #     # print("kickable is {} ".format(kickable))
    #     # print(self.action_list[team_actions[agentID]] in self.kick_actions)
    #     # print(self.num_right > 0)
    #     self.right_agent_possesion = 'N'


    #     # print("team_actions agent", team_actions[agentID])

    #     # So the changes i made to action list broke this for discretized, i think this should still work for nondiscretized

    #     # print(team_obs[agentID][self.stamina])
    #     # input()

    #     # this won't work because of the changes I made, I'd have to get the action inside of action_list somehow
    #     # i dont think i should murder myself trying to get this perfect, as long as it gets rewards for the current possible actions
    #     # just check for the value inside of action list
    #     # if self.action_list[team_actions[agentID]] in self.kick_actions and not kickable:

    #     if self.action_list[team_actions[agentID]] == 3 and not kickable:

    #         reward -= 0.1
    #         # print("agent is getting penalized for kicking when not kickable")


    #     # it looks like this is broken for discretized as well
    #     # so its not getting any rewards for kicking
    #     # print(self.action_list)
    #     # print(self.kick_actions)
    #     # input()
    #     # if self.action_list[team_actions[agentID]] in self.kick_actions and kickable:
    #     if self.action_list[team_actions[agentID]] == 3 and kickable:

    #         # if True:
    #         # if self.num_right > 0:
    #         # print(self.left_agent_possesion)
    #         if (np.array(self.left_agent_possesion) == 'N').all() and (np.array(self.right_agent_possesion) == 'N').all():
    #             # print("First Kick")
    #             reward += 1
    #             team_reward += 1.5

    #         # set initial ball position after kick
    #         if self.left_base == base:
    #             self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
    #             self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
    #         else:
    #             self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
    #             self.BR_ball_pos_y = team_obs[agentID][self.ball_y]


    #         # track ball delta in between kicks
    #         if self.left_base == base:
    #             self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
    #             self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
    #         else:
    #             self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
    #             self.BR_ball_pos_y = team_obs[agentID][self.ball_y]

    #         new_x = team_obs[agentID][self.ball_x]
    #         new_y = team_obs[agentID][self.ball_y]

    #         if self.left_base == base:
    #             ball_delta = math.sqrt((self.BL_ball_pos_x-new_x)**2+ (self.BL_ball_pos_y-new_y)**2)
    #             self.BL_ball_pos_x = new_x
    #             self.BL_ball_pos_y = new_y
    #         else:
    #             ball_delta = math.sqrt((self.BR_ball_pos_x-new_x)**2+ (self.BR_ball_pos_y-new_y)**2)
    #             self.BR_ball_pos_x = new_x
    #             self.BR_ball_pos_y = new_y

    #         self.pass_reward = ball_delta * 5.0

    #     #     ######## Pass Receiver Reward #########
    #         if self.left_base == base:
    #             if (np.array(self.left_agent_possesion) == 'L').any():
    #                 prev_poss = (np.array(self.left_agent_possesion) == 'L').argmax()
    #                 if not self.left_agent_possesion[agentID] == 'L':
    #                     self.left_passer[prev_poss] += 1 # sets passer flag to whoever passed
    #                     # Passer reward is added in step function after all agents have been checked

    #                     # reward += self.pass_reward
    #                     # team_reward += self.pass_reward
    #                     #print("received a pass worth:",self.pass_reward)
    #     #               #print('team pass reward received ')
    #     #         #Remove this check when npc ball posession can be measured
    #             if self.num_right > 0:
    #                 if (np.array(self.right_agent_possesion) == 'R').any():
    #                     enemy_possessor = (np.array(self.right_agent_possesion) == 'R').argmax()
    #                     self.right_lost_possession[enemy_possessor] -= 1.0
    #                     self.left_lost_possession[agentID] += 1.0
    #                     # print('BR lost possession')
    #                     self.pass_reward = 0

    #     #         ###### Change Possession Reward #######
    #             self.left_agent_possesion = ['N'] * self.num_left
    #             self.right_agent_possesion = ['N'] * self.num_right
    #             self.left_agent_possesion[agentID] = 'L'
    #             if possession_side != 'L':
    #                 possession_side = 'L'
    #                 #reward+=1
    #                 #team_reward+=1
    #         else:
    #             # self.opp_possession_counter[agentID] += 1
    #             if (np.array(self.right_agent_possesion) == 'R').any():
    #                 prev_poss = (np.array(self.right_agent_possesion) == 'R').argmax()
    #                 if not self.right_agent_possesion[agentID] == 'R':
    #                     self.right_passer[prev_poss] += 1 # sets passer flag to whoever passed
    #                     # reward += self.pass_reward
    #                     # team_reward += self.pass_reward
    #                     # print('opp pass reward received ')

    #             if (np.array(self.left_agent_possesion) == 'L').any():
    #                 enemy_possessor = (np.array(self.left_agent_possesion) == 'L').argmax()
    #                 self.left_lost_possession[enemy_possessor] -= 1.0
    #                 self.right_lost_possession[agentID] += 1.0
    #                 self.pass_reward = 0
    #   #             # print('BL lost possession ')

    #             self.left_agent_possesion = ['N'] * self.num_left
    #             self.right_agent_possesion = ['N'] * self.num_right
    #             self.right_agent_possesion[agentID] = 'R'
    #             if possession_side != 'R':
    #                 possession_side = 'R'
    #                 #reward+=1
    #                 #team_reward+=1

    #     ####################### reduce distance to ball - using delta  ##################
    #     # all agents rewarded for closer to ball
    #     # dist_cur = self.distance_to_ball(team_obs[agentID])
    #     # dist_prev = self.distance_to_ball(team_obs_previous[agentID])
    #     # d = (0.5)*(dist_prev - dist_cur) # if cur > prev --> +
    #     # if delta > 0:
    #     #     reward  += delta
    #     #     team_reward += delta

    #     ####################### Rewards the closest player to ball for advancing toward ball ############
    #     distance_cur, closest_agent = self.closest_player_to_ball(team_obs, num_ag)
    #     distance_prev, _ = self.closest_player_to_ball(team_obs_previous, num_ag)
    #     if agentID == closest_agent:
    #         delta = (distance_prev - distance_cur)*1.0
    #         #if delta > 0:
    #         if True:
    #             team_reward += delta
    #             reward+= delta * 5
    #             # print("distance to ball reward")
    #             # print(distance_cur, delta)
    #             pass

    #     ##################################################################################

    #     ####################### reduce ball distance to goal ##################
    #     # base left kicks
    #     r = self.ball_distance_to_goal(team_obs[agentID])
    #     r_prev = self.ball_distance_to_goal(team_obs_previous[agentID])
    #     if ((self.left_base == base) and possession_side =='L'):
    #         team_possessor = (np.array(self.left_agent_possesion) == 'L').argmax()
    #         if agentID == team_possessor:
    #             delta = (2*self.num_left)*(r_prev - r)* 1.0
    #             if True:
    #             # if delta > 0:
    #                 reward += delta * 10
    #                 team_reward += delta
    #                 # print("ball distance to goal reward.")
    #                 # pass

    #     # base right kicks
    #     elif self.right_base == base and possession_side == 'R':
    #         team_possessor = (np.array(self.right_agent_possesion) == 'R').argmax()
    #         if agentID == team_possessor:
    #             delta = (2*self.num_left)*(r_prev - r)
    #             if True:
    #             #if delta > 0:
    #                 # reward += delta
    #                 # team_reward += delta
    #                 pass
    #     # non-possessor reward for ball delta toward goal
    #     else:
    #         delta = (0*self.num_left)*(r_prev - r)
    #         if True:
    #         #if delta > 0:
    #             # reward += delta
    #             # team_reward += delta
    #             pass
    #     '''
    #         Reward agent for maximizing it's proximity to the ball
    #     '''
    #     # reward += team_obs[agentID][self.ball_proximity]

    #     # print(team_obs[agentID][self.ball_x])
    #     # print(team_obs[agentID][self.ball_y])
    #     return reward
        # rew_percent = 1.0*max(0,(self.reward_anneal - ep_num))/self.reward_anneal
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

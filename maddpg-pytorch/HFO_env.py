import random
import numpy as np
import hfo
import time
import _thread as thread
import pandas as pd
import math
import hfo_py
from HFO import get_config_path
import os, subprocess, time, signal
#from helper import *



class HFO_env():
    """HFO_env() extends the HFO environment to allow for centralized execution.

    Attributes:
        num_TA (int): Number of teammate agents. (0-11)
        num_OA (int): Number of opponent agents. (0-11)
        num_NPC (int): Number of opponent NPCs. (0-11)
        team_actions (list): List contains the current timesteps action for each
            agent. Takes value between 0 - num_states and is converted to HFO action by
            action_list.
        action_list (list): Contains the mapping from numer action value to HFO action.
        team_should_act (list of bools): Contains a boolean flag for each agent. Is
            activated to be True by Step function and becomes false when agent acts.
        team_should_act_flag (bool): Boolean flag is True if agents have
            unperformed actions, becomes False if all agents have acted.
        team_obs (list): List containing obs for each agent.
        team_obs_previous (list): List containing obs for each agent at previous timestep
        team_rewards (list): List containing reward for each agent
        start (bool): Once all agents have been launched, allows threads to listen for
            actions to take.
        world_states (list): Contains the status of the HFO world.
        team_envs (list of HFOEnvironment objects): Contains an HFOEnvironment object
            for each agent on team.
        opp_xxx attributes: Extend the same functionality for user controlled team
            to opposing team.
        loading (list of bools): Contains boolean flag for each agent. One time use
            for loading of attributes.
        loading_flag (bool): Boolean flag True if all agents have loaded.
        wait (list of bools): Contains boolean flag for each agent. Functions as a locking
            mechanism for threads
        wait_flag (bool): Boolean flag True if all agents are in sync.
        num_features (int): Number of features (varies with high or low setting)



    Todo:
        * Functionality for synchronizing team actions with opponent team actions
        """
    # class constructor
    def __init__(self, num_TA = 1,num_OA = 0,num_ONPC = 1,base = 'left',
                 goalie = False, num_trials = 10000,fpt = 100,feat_lvl = 'high',
                 act_lvl = 'low',untouched_time = 100, sync_mode = True, port = 6000,
                 offense_on_ball=0, fullstate = False, seed = 123,
                 ball_x_min = 0.3, ball_x_max = 0.5, verbose = False, log_game=False,
                 log_dir="log"):
        

        """ Initializes HFO_Env

        Args:
            num_TA (int): Number of teammate agents. (0-11)
            num_OA (int): Number of opponent agents. (0-11)
            num_ONPC (int): Number of opponent NPCs. (0-11)
            base (str): Which side for the team. ('left','right')
            goalie (bool): Should team use a goalie. (True,False)
            num_trials (int): Number of episodes
            fpt (int): Frames per trial
            feat_lvl (str): High or low feature level. ('high','low')
            act_lvl (str): High or low action level. ('high','low')



        Returns:
            HFO_Env

        """

        
        self.hfo_path = hfo_py.get_hfo_path()
        self._start_hfo_server(frames_per_trial = fpt, untouched_time = untouched_time,
                               offense_agents = num_TA, defense_agents = num_OA,
                               offense_npcs = 0, defense_npcs = num_ONPC,
                               sync_mode = sync_mode, port = port,
                               offense_on_ball = offense_on_ball,
                               fullstate = fullstate, seed = seed,
                               ball_x_min = ball_x_min, ball_x_max = ball_x_max,
                               verbose = verbose, log_game = log_game, log_dir = log_dir)                              
        self.viewer = None                      
        self.just_another_flag = True
        self.sleep_timer = 0.0000001 # sleep timer
        
        # params for low level actions
        #num_action_params = 6
        num_action_params = 5 # 2 for dash and kick 1 for turn and tackle
        self.action_params = np.asarray([[0.0]*num_action_params for i in range(num_TA)])
        if act_lvl == 'low':
            #                   pow,deg   deg       deg         pow,deg    
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo.DASH, hfo.TURN, hfo.KICK]

            self.kick_actions = [hfo.KICK] # actions that require the ball to be kickable
    
        elif act_lvl == 'high':
            self.action_list = [hfo.DRIBBLE, hfo.SHOOT, hfo.REORIENT, hfo.GO_TO_BALL, hfo.MOVE]
            self.kick_actions = [hfo.DRIBBLE, hfo.SHOOT, hfo.PASS] # actions that require the ball to be kickable
        
        self.num_TA = num_TA
        self.num_OA = num_OA
        self.num_ONPC = num_ONPC

        self.act_lvl = act_lvl
        self.feat_lvl = feat_lvl
        
        if feat_lvl == 'low':
            self.num_features = 50 + 9*num_TA + 9*num_OA + 9*num_ONPC
        elif feat_lvl == 'high':
            self.num_features = (6*num_TA) + (3*num_OA) + (3*num_ONPC) + 6


        # Create env for each teammate
        self.team_envs = [hfo.HFOEnvironment() for i in range(num_TA)]
        self.opp_team_envs = [hfo.HFOEnvironment() for i in range(num_OA)]

        # flag that says when the episode is done
        self.d = False

        # flag to wait for all the agents to load
        self.start = False
        # loading flag
        self.loading_flag = True
        # array to hold the loading flag state for each agent
        self.loading = np.array([1]*num_TA)
        # talks to the agents
        self.wait = np.array([0]*num_TA)
        self.wait_flag = False


        # Initialization of mutable lists to be passsed to threads
        # action each team mate is supposed to take when its time to act
        self.team_actions = np.array([2]*num_TA)

        # each individual agent has a wait flag that keeps the agent from doing the next action
        self.team_should_act = np.array([0]*num_TA)
        # signals all team agents to act
        self.team_should_act_flag = False

        # observation space for all team mate agents
        self.team_obs = np.empty([num_TA,self.num_features],dtype=float)
        # previous state for all team agents
        self.team_obs_previous = np.empty([num_TA,self.num_features],dtype=float)

        # reward for each agent
        self.team_rewards = np.zeros(num_TA)

        # not used yet
        self.opp_actions = np.zeros(num_OA)
        # not used yet
        self.opp_should_act = np.array([0]*num_OA)
        #not used yet
        self.opp_should_act_flag = False

        #self.opp_obs = np.empty([num_OA,5],dtype=object)
        #self.opp_rewards = np.zeros(num_OA)

        # keeps track of world state
        self.world_status = 0

        # Create thread for each teammate
        for i in range(num_TA):
            print("Connecting player %i" % i , "on team %s to the server" % base)
            thread.start_new_thread(self.connect,(feat_lvl, base,
                                             False,i,fpt,act_lvl,))
            time.sleep(0.5)


        # Create thread for each opponent (set one to goalie)
        if base == 'left':
            opp_base = 'right'
        elif base == 'right':
            opp_base = 'left'

#         for i in range(num_OA):
#             if i==0:
#                 thread.start_new_thread(connect,(self, feat_lvl, opp_base,
#                                                  True,i,))
#             else:
#                 thread.start_new_thread(connect,(self, feat_lvl, opp_base,
#                                                  False,i,))
#             time.sleep(1)

        print("All players connected to server")
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
            return self.team_obs[agent_id]
        elif side == 'opp':
            return self.opp_obs[agent_id]

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


    def Step(self,actions,side,params=[]):
        """ Performs each agents' action from actions and returns tuple (obs,rewards,world_status)

        Args:
            actions (list of ints); List of integers corresponding to the action each agent will take
            side (str): Which team agent belongs to. ('team', 'opp')

        Returns:
            Status of HFO World

        Todo:
            * Add functionality for opp team

        """
        [self.Queue_action(i,actions[i],side,params) for i in range(len(actions))]

        return np.asarray(self.team_obs),self.team_rewards,self.d, self.team_envs[0].statusToString(self.world_status)



    def Queue_action(self,agent_id,action,side,params=[]):
        """ Queues an action on agent, and if all agents have received action instructions performs the actions.

        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            side (str): Which team agent belongs to. ('team', 'opp')


        Todo:
            * Add halting until agents from both teams have received action

        """
        # Function is running too fast for agents to act so wait until the agents have acted

        while not self.wait_flag:
            time.sleep(self.sleep_timer)
        self.wait[agent_id] = False

        if side == 'team':
            self.team_actions[agent_id] = action
            if self.act_lvl == 'low':
                for p in range(params.shape[1]):
                    self.action_params[agent_id][p] = params[agent_id][p]
            
        elif side == 'opp':
            self.opp_actions[agent_id] = action

        # Sets action flag for this agent to True.
        # Sets team action flag to True allowing threads to take action only
        #once each thread's personal action flag is set to true.
        self.team_should_act[agent_id] = True
        if self.team_should_act.all():
            self.team_should_act_flag = True
            while self.team_should_act_flag:
                time.sleep(self.sleep_timer)

        if not self.wait.any():
            self.wait_flag = False
            self.just_another_flag = False
        

        #time.sleep(self.sleep_timer) ### *** without this sleep function the process crashes. specifically, 0.001


#################################################
######################  Utils for Reward ########
#################################################

    def get_kickable_status(self,agentID,obs):
        ball_kickable = False
        if self.feat_lvl == 'high':
            if obs[agentID][5] == 1:
                ball_kickable = True
        elif self.feat_lvl == 'low':
            if obs[agentID][12] == 1:
                ball_kickable = True
        return ball_kickable
            
            

    def apprx_to_goal(self, obs):
        # return the proximity of the agent to the goal center 
        if self.feat_lvl == 'high':            
            return obs[6]
        else:
            return obs[15]

    # low level feature (1 for closest to object -1 for furthest)
    def ball_proximity(self,obs):
        if obs[50]: # ball pos valid
            return obs[53]
        else:
            print('no ball...')
            return -1
        
        
    def ball_distance_to_goal(self,obs):
        if self.feat_lvl == 'high': 
            relative_x = 1 - obs[3]
            relative_y = 0 - obs[4]
            #Returns the relative distance between the goal and the ball
            ball_distance_to_goal = math.sqrt(relative_x**2+relative_y**2)

        elif self.feat_lvl =='low':
            relative_x = 0
            relative_y = 0
            ball_proximity = obs[53]
            goal_proximity = obs[15]
            ball_dist = 1.0 - ball_proximity
            goal_dist = 1.0 - goal_proximity
            kickable = obs[12]
            ball_ang_sin_rad = obs[51]
            ball_ang_cos_rad = obs[52]
            ball_ang_rad = math.acos(ball_ang_cos_rad)
            if ball_ang_sin_rad < 0:
                ball_ang_rad *= -1.
            goal_ang_sin_rad = obs[13]
            goal_ang_cos_rad = obs[14]
            goal_ang_rad = math.acos(goal_ang_cos_rad)
            if goal_ang_sin_rad < 0:
                goal_ang_rad *= -1.
            alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
            ball_distance_to_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                                       2.*ball_dist*goal_dist*math.cos(alpha))
    
        return ball_distance_to_goal, relative_x, relative_y
    
 
    #Finds agent distance to ball - high level feature
    def distance_to_ball(self, obs):
        #Relative x and y is the offset between the ball and the agent.
        if self.feat_lvl == 'high':
            relative_x = obs[0]-obs[3]
            relative_y = obs[1]-obs[4]
            ball_distance = math.sqrt(relative_x**2+relative_y**2)
        
        return ball_distance, relative_x, relative_y
    
    # takes param index (0-4)
    def get_valid_scaled_param(self,agentID,param):
        if param == 0: # dash power
            return ((self.action_params[agentID][0].clip(-1,1) + 1)/2)*100 # only allows for positive dash power, change later
        elif param == 1: # dash deg
            return self.action_params[agentID][1].clip(-1,1)*180
        elif param == 2: # turn deg
            return self.action_params[agentID][2].clip(-1,1)*180
        # tackle deg
        elif param == 3: # kick power
            return ((self.action_params[agentID][3].clip(-1,1) + 1)/2)*100
        elif param == 4: # kick deg
            return self.action_params[agentID][4].clip(-1,1)*180

        # with tackle


    def get_excess_param_distance(self,agentID):
        distance = 0
        distance += (self.action_params[agentID][0].clip(-1,1) - self.action_params[agentID][0])**2
        distance += (self.action_params[agentID][1].clip(-1,1) - self.action_params[agentID][1])**2
        distance += (self.action_params[agentID][2].clip(-1,1) - self.action_params[agentID][2])**2
        distance += (self.action_params[agentID][3].clip(-1,1) - self.action_params[agentID][3])**2
        distance += (self.action_params[agentID][4].clip(-1,1) - self.action_params[agentID][4])**2

        return distance

    # Engineered Reward Function
    def getReward(self,s,agentID):
        reward=0
        #---------------------------

        ########################### keep the ball kickable ####################################
        #team_kickable = False
        #team_kickable = np.array([self.get_kickable_status(i,self.team_obs_previous) for i in range(self.num_TA)]).any() # kickable by team
        #if team_kickable :
        #    reward+= 1
        
        if self.action_list[self.team_actions[agentID]] in self.kick_actions and self.get_kickable_status(agentID,self.team_obs_previous): # uses action just performed, with previous obs, (both at T)
            reward+= 1 # kicked when avaialable; I am still concerend about the timeing of the team_actions and the kickable status
           
        # out of bounds penalty
        #if self.team_obs[agentID][46] > .99 or self.team_obs[agentID][47] > .99 or self.team_obs[agentID][48] > .99  or self.team_obs[agentID][49] > .99:
        #    reward += -.1
        
        
        # add a penalty for rnning out of stamina?
        

        ####################### penalty based on sum of square distances of excess params ##############
        #reward += -self.get_excess_param_distance(agentID)*0.1
        
        ####################### penalty for invalid action  ###############################
        if self.feat_lvl == 'high':        
                  if self.team_obs[agentID][-2] == -1:
                        reward+= -1
        elif self.feat_lvl == 'low':        
                  if  self.team_obs[agentID][-1] != 1:
                        reward+= -1
                        
        ####################### reduce distance to ball - using delta  ##################
        if self.feat_lvl == 'high':
            r,_,_ = self.distance_to_ball(self.team_obs[agentID])
            r_prev,_,_ = self.distance_to_ball(self.team_obs_previous[agentID])
            reward += (r_prev - r) # if [prev > r] ---> positive; if [r > prev] ----> negative
        elif self.feat_lvl == 'low':
            prox_cur = self.ball_proximity(self.team_obs[agentID])
            prox_prev = self.ball_proximity(self.team_obs_previous[agentID])
            reward   += prox_cur - prox_prev # if cur > prev --> +     
        ##################################################################################
        
        ####################### reduce ball distance to goal - using delta  ##################
        r,_,_ = self.ball_distance_to_goal(self.team_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        r_prev,_,_ = self.ball_distance_to_goal(self.team_obs_previous[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        reward += (3)*(r_prev - r)
        ##################################################################################
        
        if s=='Goal':
            reward+=10
        #---------------------------
        #elif s=='CapturedByDefense':
        #    reward+=-100
        #---------------------------
        #elif s=='OutOfBounds':
        #    reward+=-1
        #---------------------------
        #Cause Unknown Do Nothing
        #elif s=='OutOfTime':
        #    reward+=-100
        #---------------------------
        #elif s=='InGame':
        #    reward+=0
        #---------------------------
        #elif s=='SERVER_DOWN':
        #    reward+=0
        #---------------------------
        #else:
        #    print("Error: Unknown GameState", s)
        #    reward = -1
        return reward


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self,feat_lvl, base, goalie, agent_ID,fpt,act_lvl):
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

        config_dir=get_config_path() 
        #config_dir = '/Users/sajjadi/Desktop/work/HFO/bin/teams/base/config/formations-dt'
        recorder_dir = 'log/'
        self.team_envs[agent_ID].connectToServer(feat_lvl, config_dir=config_dir,
                            server_port=6000, server_addr='localhost', team_name=base, 
                                                 play_goalie=goalie,record_dir =recorder_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Once all agents have been loaded,
        # wait for action command, take action, update: obs, reward, and world status
        while(True):
            while(self.start):
                j = 0 # j to maximum episode length
                d = False
                self.team_obs_previous[agent_ID] = self.team_envs[agent_ID].getState() # Get initial state
                self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # Get initial state
                
                while j < fpt:
                    # If the action flag is set, each thread takes its action
                    # Sets its personal flag to false, such that if all agents have taken their action
                    # and have updated values we may return to the Step function call
                    # (This synchronizes all agents actions so that they halt until other agents have taken theirs)


                    self.wait[agent_ID] = True
                    while not self.wait_flag:
                        if(self.wait.all()):
                            self.wait_flag = True
                        else:
                            time.sleep(self.sleep_timer)

                    if self.loading[agent_ID]:
                        print("Agent %i loaded" % agent_ID)
                        self.loading[agent_ID] = False
                    if (not self.loading.any()) and self.loading_flag:
                        self.loading_flag = False
                        print("Loaded and ready for action order")




                    while not self.team_should_act_flag and not self.team_should_act[agent_ID]:
                        time.sleep(self.sleep_timer)
                    
                    # take the action
                    if act_lvl == 'high':
                        self.team_envs[agent_ID].act(self.action_list[self.team_actions[agent_ID]]) # take the action
                    elif act_lvl == 'low':
                        # use params for low level actions
                        
                        # scale action params
                        a = self.team_actions[agent_ID]
                        
                        # with tackle -- outdated
                        """if a == 0:
                            self.team_envs[agent_ID].act(self.action_list[a],self.action_params[agent_ID][0],self.action_params[agent_ID][1])
                            #print(self.action_list[a],self.action_params[agent_ID][0],self.action_params[agent_ID][1])
                        elif a == 1:
                            #print(self.action_list[a],self.action_params[agent_ID][2])
                            self.team_envs[agent_ID].act(self.action_list[a],self.action_params[agent_ID][2])
                        elif a == 2:
                            #print(self.action_list[a],self.action_params[agent_ID][3])                            
                            self.team_envs[agent_ID].act(self.action_list[a],self.action_params[agent_ID][3])           
                        elif a ==3:
                            #print(self.action_list[a],self.action_params[agent_ID][4],self.action_params[agent_ID][5])
                            self.team_envs[agent_ID].act(self.action_list[a],self.action_params[agent_ID][4],self.action_params[agent_ID][5])
                        """
                        
                        # without tackle
                        if a == 0:
                            self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,0),self.get_valid_scaled_param(agent_ID,1))
                            #print(self.action_list[a],self.action_params[agent_ID][0],self.action_params[agent_ID][1])
                        elif a == 1:
                            #print(self.action_list[a],self.action_params[agent_ID][2])
                            self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,2))                       
                        elif a ==2:
                            #print(self.action_list[a],self.action_params[agent_ID][4],self.action_params[agent_ID][5])
                            self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,3),self.get_valid_scaled_param(agent_ID,4))                                                    

                    self.team_should_act[agent_ID] = False # Set personal action flag as done

                    # Halts until all threads have taken their action and resets flag to off
                    while(self.team_should_act.any()):
                        time.sleep(self.sleep_timer)
                        
                    self.team_obs_previous[agent_ID] = self.team_obs[agent_ID]
                    self.world_status = self.team_envs[agent_ID].step() # update world
                 
                    self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # update obs after all agents have acted
                    self.team_rewards[agent_ID] = self.getReward(
                    self.team_envs[agent_ID].statusToString(self.world_status),agent_ID) # update reward
                    self.team_should_act_flag = False

                    if self.world_status == hfo.IN_GAME:
                        self.d = False
                    else:
                        self.d = True

                    while self.just_another_flag:
                        time.sleep(self.sleep_timer)
                    self.just_another_flag = True

                    # Break if episode done
                    j+=1
                    

                    if self.d == True:
                        break


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # found from https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    def _start_hfo_server(self, frames_per_trial=100,
                              untouched_time=100, offense_agents=1,
                              defense_agents=0, offense_npcs=0,
                              defense_npcs=0, sync_mode=True, port=6000,
                              offense_on_ball=0, fullstate=False, seed=123,
                              ball_x_min=0.0, ball_x_max=0.2,
                              verbose=False, log_game=False,
                              log_dir="log"):
            """
            Starts the Half-Field-Offense server.
            frames_per_trial: Episodes end after this many steps.
            untouched_time: Episodes end if the ball is untouched for this many steps.
            offense_agents: Number of user-controlled offensive players.
            defense_agents: Number of user-controlled defenders.
            offense_npcs: Number of offensive bots.
            defense_npcs: Number of defense bots.
            sync_mode: Disabling sync mode runs server in real time (SLOW!).
            port: Port to start the server on.
            offense_on_ball: Player to give the ball to at beginning of episode.
            fullstate: Enable noise-free perception.
            seed: Seed the starting positions of the players and ball.
            ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
            verbose: Verbose server messages.
            log_game: Enable game logging. Logs can be used for replay + visualization.
            log_dir: Directory to place game logs (*.rcg).
            """
            self.server_port = port
            cmd = self.hfo_path + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --log-dir %s --record"\
                  % (frames_per_trial, untouched_time, offense_agents,
                     defense_agents, offense_npcs, defense_npcs, port,
                     offense_on_ball, seed, ball_x_min, ball_x_max,
                     log_dir)
            if not sync_mode: cmd += " --no-sync"
            if fullstate:     cmd += " --fullstate"
            if verbose:       cmd += " --verbose"
            if not log_game:  cmd += " --no-logging"
            print('Starting server with command: %s' % cmd)
            self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
            time.sleep(3) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        
        
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

import random
import numpy as np
import time
# import _thread as thread
import threading
import pandas as pd
import math
from HFO.hfo import hfo
from HFO import get_config_path, get_hfo_path, get_viewer_path
import os, subprocess, time, signal
#from helper import *



possession_side = 'N'

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
    Todo:
        * Functionality for synchronizing team actions with opponent team actions
        * Add the ability for team agents to function with npcs taking the place of opponents
        """
    # class constructor
    def __init__(self, num_TNPC=0,num_TA = 1,num_OA = 0,num_ONPC = 1,base = 'base_left',
                 goalie = False, num_trials = 10000,fpt = 100,feat_lvl = 'high',
                 act_lvl = 'low',untouched_time = 100, sync_mode = True, port = 6000,
                 offense_on_ball=0, fullstate = False, seed = 123,
                 ball_x_min = -0.8, ball_x_max = 0.8, ball_y_min = -0.8, ball_y_max = 0.8,
                 verbose = False, rcss_log_game=False, hfo_log_game=False, log_dir="log",team_rew_anneal_ep=1000,
                 agents_x_min=-0.8, agents_x_max=0.8, agents_y_min=-0.8, agents_y_max=0.8,
                 change_every_x=5, change_agents_x=0.1, change_agents_y=0.1, change_balls_x=0.1,
                 change_balls_y=0.1, control_rand_init=False,record=True,
                 defense_team_bin='helios15', offense_team_bin='helios16', run_server=False, deterministic=True):

        """ Initializes HFO_Env
        Args:
            num_TA (int): Number of teammate agents. (0-11)
            num_OA (int): Number of opponent agents. (0-11)
            num_ONPC (int): Number of opponent NPCs. (0-11)
            base (str): Which side for the team. ('base_left','base_right') NOTE: Keep this at base_left
            goalie (bool): Should team use a goalie. (True,False)
            num_trials (int): Number of episodes
            fpt (int): Frames per trial
            feat_lvl (str): High or low feature level. ('high','low')
            act_lvl (str): High or low action level. ('high','low')
        Returns:
            HFO_Env
        """
        self.pass_reward = 0.0
        self.agent_possession_team = ['N'] * num_TA
        self.team_passer = [0]*num_TA
        self.opp_passer = [0]*num_TA
        self.team_lost_possession = [0]*num_TA
        self.opp_lost_possession = [0]*num_TA
        self.agent_possession_opp = ['N'] * num_OA
        self.team_possession_counter = [0] * num_TA
        self.opp_possession_counter = [0] * num_OA
        self.goalie = goalie
        self.team_rew_anneal_ep = team_rew_anneal_ep
        self.port = port
        self.hfo_path = get_hfo_path()
        if run_server:
            self._start_hfo_server(frames_per_trial = fpt, untouched_time = untouched_time,
                                    offense_agents = num_TA, defense_agents = num_OA,
                                    offense_npcs = num_TNPC, defense_npcs = num_ONPC,
                                    sync_mode = sync_mode, port = port,
                                    offense_on_ball = offense_on_ball,
                                    fullstate = fullstate, seed = seed,
                                    ball_x_min = ball_x_min, ball_x_max = ball_x_max,
                                    ball_y_min= ball_y_min, ball_y_max= ball_y_max,
                                    verbose = verbose, rcss_log_game = rcss_log_game, 
                                    hfo_log_game=hfo_log_game, log_dir = log_dir,
                                    agents_x_min=agents_x_min, agents_x_max=agents_x_max,
                                    agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                                    change_every_x=change_every_x, change_agents_x=change_agents_x,
                                    change_agents_y=change_agents_y, change_balls_x=change_balls_x,
                                    change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=record,
                                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, deterministic=deterministic)

        self.viewer = None
        # self.sleep_timer = 1.0 # sleep timer
        # self.sleep_timer2 = 15.0
        # params for low level actions
        #num_action_params = 6
        num_action_params = 5 # 2 for dash and kick 1 for turn and tackle
        self.team_action_params = np.asarray([[0.0]*num_action_params for i in range(num_TA)])
        self.opp_action_params = np.asarray([[0.0]*num_action_params for i in range(num_OA)])
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

        self.base = base
        self.fpt = fpt
        self.been_kicked_team = False
        self.been_kicked_opp = False
        self.act_lvl = act_lvl
        self.feat_lvl = feat_lvl
        self.team_possession = False
        self.opp_possession = False
        if feat_lvl == 'low':
            self.team_num_features = 59 + 13*(num_TA-1) + 12*num_OA + 4 + 9*num_ONPC + 1 + 2 + 1
            self.opp_num_features = 59 + 13*(num_OA-1) + 12*num_TA + 4 + 9*num_ONPC + 1 + 2 + 1
        elif feat_lvl == 'high':
            self.team_num_features = (6*num_TA) + (3*num_OA) + (3*num_ONPC) + 6
            self.opp_num_features = (6*num_OA) + (3*num_TA) + (3*num_ONPC) + 6 
        self.open_goal = 58
        self.team_goal_angle_beg = 59
        self.team_goal_angle_end = self.team_goal_angle_beg +(num_TA -1)
        self.opp_goal_angle_beg = self.team_goal_angle_end
        self.opp_goal_angle_end = self.opp_goal_angle_beg + num_TA
        self.team_pass_angle_beg = self.opp_goal_angle_end 
        self.team_pass_angle_end = self.team_pass_angle_beg + num_TA - 1
        self.team_unif_beg = -(2*num_TA) -(2*(num_TA)) - (2*num_OA) - 2 -2 - 1
        self.team_unif_end = -(2*num_TA) + num_TA - 1 -(2*(num_TA-1)) - (2*num_TA) - 2 -2 - 1
        # 58 = OPEN GOAL
        # 59:59+(num_TA - 1) = teammates goal angle
        # 59 + (num_TA): 59+ num_TA +(num_TA-1) = opp goal angle
        # 59+ (2*num_TA) - 1: 59+(2*num_TA)-1 + (num_TA-1) = team pass angle

        # Create env for each teammate
        self.team_envs = [hfo.HFOEnvironment() for i in range(num_TA)]
        self.opp_team_envs = [hfo.HFOEnvironment() for i in range(num_OA)]

        # flag that says when the episode is done
        # Use number for tensor
        self.d = 0

        # flag to wait for all the agents to load
        self.start = False

        # Locks to cause agents to wait for each other
        # self.wait_for_queue = True
        # self.wait_for_connect_vals = False

        # Counters to keep agents on the same page in the connect function
        # Each cell represents a corresponding counter to an agent
        # Need to have counters that are not shared among the agents ergo
        # need to have a counter per agent
        # self.sync_after_queue_team = np.zeros(num_TA)
        # self.sync_after_queue_opp = np.zeros(num_OA)
        # self.sync_before_step_team = np.zeros(num_TA)
        # self.sync_before_step_opp = np.zeros(num_OA)
        # self.sync_at_status_team = np.zeros(num_TA)
        # self.sync_at_status_opp = np.zeros(num_OA)
        # self.sync_at_reward_team = np.zeros(num_TA)
        # self.sync_at_reward_opp = np.zeros(num_OA)

        self.sync_after_queue = threading.Barrier(num_TA+num_OA+1)
        self.sync_before_step = threading.Barrier(num_TA+num_OA+1)
        self.sync_at_status = threading.Barrier(num_TA+num_OA)
        self.sync_at_reward = threading.Barrier(num_TA+num_OA)

        # Initialization of mutable lists to be passsed to threads
        # action each team mate is supposed to take when its time to act
        self.team_actions = np.array([2]*num_TA)
        # observation space for all team mate agents
        self.team_obs = np.empty([num_TA,self.team_num_features],dtype=float)
        # previous state for all team agents
        self.team_obs_previous = np.empty([num_TA,self.team_num_features],dtype=float)
        # reward for each agent
        self.team_rewards = np.zeros(num_TA)

        self.opp_actions = np.array([2]*num_OA)
        # observation space for all team mate agents
        self.opp_team_obs = np.empty([num_OA,self.opp_num_features],dtype=float)
        # previous state for all team agents
        self.opp_team_obs_previous = np.empty([num_OA,self.opp_num_features],dtype=float)
        # reward for each agent
        self.opp_rewards = np.zeros(num_OA)

        # keeps track of world state
        self.world_status = 0
        
        self.team_base = base
        self.opp_base = ''

        # Create thread for each opponent (set one to goalie)
        if base == 'base_left':
            self.opp_base = 'base_right'
        elif base == 'base_right':
            self.opp_base = 'base_left'

        
    def launch(self):
        # Create thread for each teammate
        for i in range(self.num_TA):
            if i == 0:
                print("Connecting player %i" % i , "on team %s to the server" % self.base)
                # thread.start_new_thread(self.connect,(self.port,self.feat_lvl, self.base,
                #                                 self.goalie,i,self.fpt,self.act_lvl,))
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.base,
                                                self.goalie,i,self.fpt,self.act_lvl,))
                t.start()
            else:
                print("Connecting player %i" % i , "on team %s to the server" % self.base)
                # thread.start_new_thread(self.connect,(self.port,self.feat_lvl, self.base,
                #                                 False,i,self.fpt,self.act_lvl,))
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.base,
                                                False,i,self.fpt,self.act_lvl,))
                t.start()
            time.sleep(1.5)
        
        for i in range(self.num_OA):
            if i == 0:
                print("Connecting player %i" % i , "on Opponent %s to the server" % self.opp_base)
                # thread.start_new_thread(self.connect,(self.port,self.feat_lvl, self.opp_base,
                #                                 self.goalie,i,self.fpt,self.act_lvl,))
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.opp_base,
                                                self.goalie,i,self.fpt,self.act_lvl,))
                t.start()
            else:
                print("Connecting player %i" % i , "on Opponent %s to the server" % self.opp_base)
                # thread.start_new_thread(self.connect,(self.port,self.feat_lvl, self.opp_base,
                #                                 False,i,self.fpt,self.act_lvl,))
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.opp_base,
                                                False,i,self.fpt,self.act_lvl,))
                t.start()

            time.sleep(1.5)
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
            return self.opp_team_obs[agent_id]

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


    def Step(self, team_actions, opp_actions, team_params=[], opp_params=[]):
        """ Performs each agents' action from actions and returns tuple (obs,rewards,world_status)
        Args:
            actions (list of ints); List of integers corresponding to the action each agent will take
            side (str): Which team agent belongs to. ('team', 'opp')
        Returns:
            Status of HFO World
        Todo:
            * Add functionality for opp team
        """
        # Queue actions for team
        [self.Queue_action(i,self.team_base,team_actions[i],team_params) for i in range(len(team_actions))]
        # Queue actions for opposing team
        [self.Queue_action(j,self.opp_base,opp_actions[j],opp_params) for j in range(len(opp_actions))]

        # self.sync_after_queue_team += 1
        # self.sync_after_queue_opp += 1
        # while (self.sync_after_queue_team.sum() + self.sync_after_queue_opp.sum()) % ((self.num_TA + self.num_OA) * 2) != 0:
        #     time.sleep(self.sleep_timer)
        self.sync_after_queue.wait()
            
        # self.wait_for_queue = False
        # self.wait_for_connect_vals = True

        # while self.wait_for_connect_vals:
        #     time.sleep(self.sleep_timer)
        self.sync_before_step.wait()
        #print('Actions, Obs, rewards, and status ready')

        self.team_rewards = [rew + self.pass_reward if passer else rew for rew,passer in zip(self.team_rewards,self.team_passer)]

        self.opp_rewwards =[rew + self.pass_reward if passer else rew for rew,passer in zip(self.opp_rewards,self.opp_passer)]
        

        self.team_rewards = np.add( self.team_rewards, self.team_lost_possession)

        self.opp_rewards = np.add(self.opp_rewards, self.opp_lost_possession)

        #team_rew -=  self.team_lost_possession
        #opp_rew -=  self.opp_lost_possession
        # rew + 1 list comprehension adds the reward for passer if pass was received
        self.team_passer = [0]*self.num_TA
        self.opp_passer = [0]*self.num_TA
        self.team_lost_possession = [0]*self.num_TA
        self.opp_lost_possession = [0]*self.num_TA
        return np.asarray(self.team_obs),self.team_rewards,np.asarray(self.opp_team_obs),self.opp_rewards, \
                self.d, self.world_status



    def Queue_action(self,agent_id,base,action,params=[]):
        """ Queues an action on agent, and if all agents have received action instructions performs the actions.
        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            base (str): Which side an agent belongs to. ('base_left', 'base_right')
        """

        if self.team_base == base:
            self.team_actions[agent_id] = action
            if self.act_lvl == 'low':
                for p in range(params.shape[1]):
                    self.team_action_params[agent_id][p] = params[agent_id][p]
        else:
            self.opp_actions[agent_id] = action
            if self.act_lvl == 'low':
                for p in range(params.shape[1]):
                    self.opp_action_params[agent_id][p] = params[agent_id][p]


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
    
    def get_agent_possession_status(self,agentID,base):
        if self.team_base == base:
            if self.agent_possession_team[agentID] == 'L':
                self.team_possession_counter[agentID] += 1
            
            return self.team_possession_counter[agentID]
        else:
            if self.agent_possession_opp[agentID] == 'R':
                self.opp_possession_counter[agentID] += 1
            
            return self.opp_possession_counter[agentID]  

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
        
    def closest_player_to_ball(self, team_obs, num_agents):
        '''
        teams receive reward based on the distance of their closest agent to the ball
        '''
        closest_player_index = 0
        ball_prox = self.ball_proximity(team_obs[0])
        for i in range(1, num_agents):
            temp_prox = self.ball_proximity(team_obs[i])
            if ball_prox < temp_prox:
                closest_player_index = i
                ball_prox = temp_prox
        
        return ball_prox, closest_player_index
    
    def agent_possession_reward(self,base,agentID):
        '''
        agent receives reward for possessing ball
        '''
        rew_amount = 0.001
        if self.team_base == base:
            if self.agent_possession_team[agentID] == 'L':
                return rew_amount
        else:
            if self.agent_possession_opp[agentID] == 'R':
                return rew_amount
        return 0.0
                
    def unnormalize_unif(self,unif):
        return int(np.round(unif * 100,decimals=0))
            
    def team_possession_reward(self,base):

        '''
        teams receive reward based on possession defined by which side had the ball kickable last
        '''
        rew_amount = 0.001
        global possession_side
        if self.team_base == base:
            if  possession_side == 'L':
                    return rew_amount
            if  possession_side == 'R':
                    return -rew_amount
        else: 
            if  possession_side == 'R':
                    return rew_amount
            if  possession_side == 'L':
                    return -rew_amount
        
        return 0.0
        
        
    def ball_distance_to_goal(self,obs):
        if self.feat_lvl == 'high': 
            relative_x = 0.84 - obs[3]
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
    def get_valid_scaled_param(self,agentID,param,base):
        if self.team_base == base:
            self.action_params = self.team_action_params
        else:
            self.action_params = self.opp_action_params

        if param == 0: # dash power
            #return ((self.action_params[agentID][0].clip(-1,1) + 1)/2)*100
            return self.action_params[agentID][0].clip(-1,1)*100
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


    def get_excess_param_distance(self,agentID,base):
        if self.team_base == base:
            self.action_params = self.team_action_params
        else:
            self.action_params = self.opp_action_params

        distance = 0
        distance += (self.action_params[agentID][0].clip(-1,1) - self.action_params[agentID][0])**2
        distance += (self.action_params[agentID][1].clip(-1,1) - self.action_params[agentID][1])**2
        distance += (self.action_params[agentID][2].clip(-1,1) - self.action_params[agentID][2])**2
        distance += (self.action_params[agentID][3].clip(-1,1) - self.action_params[agentID][3])**2
        distance += (self.action_params[agentID][4].clip(-1,1) - self.action_params[agentID][4])**2

        return distance

    
    def getPretrainRew(self,s,d,base):
        
        reward=0.0
        team_reward = 0.0
        #---------------------------

        if d:
            if self.team_base == base:
            # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----
                if s==1: # Goal left
                    reward+=10
                elif s==2: # Goal right
                    reward+=-10
                elif s==3: # OOB
                    reward+=-0.2
                #---------------------------
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
            else:
                if s==1: # Goal left
                    reward+=-10
                elif s==2: # Goal right
                    reward+=+10
                elif s==3: # OOB
                    reward+=-0.2
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
        return reward

    def unnormalize(self,val):
        return (val +1.0)/2.0
    
    # Engineered Reward Function
    #   To-do: Add the ability of team agents to know if opponent npcs hold ball posession. For now, having npc opponent disables first kick award
    def getReward(self,s,agentID,base,ep_num):
        reward=0.0
        team_reward = 0.0
        goal_points = 25.0
        #---------------------------
        global possession_side
        if self.d:
            if self.team_base == base:
            # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----
                if s=='Goal_By_Left' and self.agent_possession_team[agentID] == 'L':
                    reward+= goal_points
                elif s=='Goal_By_Left':
                    reward+= goal_points/10.0 # teammates get 10% of points
                elif s=='Goal_By_Right':
                    reward+=-goal_points
                elif s=='OutOfBounds':
                    reward+=-0.2

                possession_side = 'N' # at the end of each episode we set this to none
                self.agent_possession_team = ['N'] * self.num_TA
                return reward
            else:
                if s=='Goal_By_Right' and self.agent_possession_opp[agentID] == 'R':
                    reward+=goal_points
                elif s=='Goal_By_Right':
                    reward+=goal_points/10.0
                elif s=='Goal_By_Left':
                    reward+=-goal_points
                elif s=='OutOfBounds':
                    reward+=-0.2

                possession_side = 'N'
                self.agent_possession_opp = ['N'] * self.num_OA
                return reward
        

        # set global possessor flag     
        # If anyone kicked the ball, on left get which one
        kicked = np.array([self.action_list[self.team_actions[i]] in self.kick_actions and self.get_kickable_status(i,self.team_obs_previous) for i in range(self.num_TA)])
        if kicked.any():
            self.team_obs[:,-2] = (kicked.argmax() + 1)/100.0
        else:
            self.team_obs[:,-2] = 0
        # If anyone kicked the ball on right
        kicked = np.array([self.action_list[self.opp_actions[i]] in self.kick_actions and self.get_kickable_status(i,self.opp_team_obs_previous) for i in range(self.num_TA)])
        if kicked.any():
            self.opp_team_obs[:,-1] = (kicked.argmax() + 1)/100.0
        else:
            self.opp_team_obs[:,-1] = 0

        
        if self.team_base == base:
            team_actions = self.team_actions
            team_obs = self.team_obs
            team_obs_previous = self.team_obs_previous
            opp_obs = self.opp_team_obs
            opp_obs_previous = self.opp_team_obs_previous
            num_ag = self.num_TA
        else:
            team_actions = self.opp_actions
            team_obs = self.opp_team_obs
            team_obs_previous = self.opp_team_obs_previous
            opp_obs = self.team_obs
            opp_obs_previous = self.team_obs_previous
            num_ag = self.num_OA

        if team_obs[agentID][7] < 0 : # LOW STAMINA
            reward -= 0.05
            team_reward -= 0.05
            # print ('low stamina')
        


        ############ Kicked Ball #################
        if self.action_list[team_actions[agentID]] in self.kick_actions and self.get_kickable_status(agentID,team_obs_previous):            
            #Remove this check when npc ball posession can be measured
            if self.num_OA > 0:
                if (np.array(self.agent_possession_team) == 'N').all() and (np.array(self.agent_possession_opp) == 'N').all():
                    #print("First Kick")
                    reward += 10.0
                    team_reward +=10.0
                # set initial ball position after kick
                    self.ball_pos_x = self.team_envs[0].getBallX()/52.5
                    self.ball_pos_y = self.team_envs[0].getBallY()/34.0

            # track ball delta in between kicks
            new_x = self.team_envs[0].getBallX()/52.5
            new_y = self.team_envs[0].getBallY()/34.0
            self.ball_delta = math.sqrt((self.ball_pos_x-new_x)**2+ (self.ball_pos_y-new_y)**2)
            self.ball_pos_x = new_x
            self.ball_pos_y = new_y
            self.pass_reward = self.ball_delta * 15.0


            ######## Pass Receiver Reward #########
            if self.team_base == base:
                if (np.array(self.agent_possession_team) == 'L').any():
                    prev_poss = (np.array(self.agent_possession_team) == 'L').argmax()
                    if not self.agent_possession_team[agentID] == 'L':
                        self.team_passer[prev_poss] += 1 # sets passer flag to whoever passed
                        # Passer reward is added in step function after all agents have been checked
                        
                        reward += self.pass_reward
                        team_reward += self.pass_reward
                        #print("received a pass worth:",self.pass_reward)
                        #print('team pass reward received ')
                #Remove this check when npc ball posession can be measured
                if self.num_OA > 0:
                    if (np.array(self.agent_possession_opp) == 'R').any():
                        enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
                        self.opp_lost_possession[enemy_possessor] -= 10.0
                        self.team_lost_possession[agentID] += 10.0
                        # print('opponent lost possession')

                ###### Change Possession Reward #######
                self.agent_possession_team = ['N'] * self.num_TA
                self.agent_possession_opp = ['N'] * self.num_OA
                self.agent_possession_team[agentID] = 'L'
                if possession_side != 'L':
                    possession_side = 'L'    
                    #reward+=1
                    #team_reward+=1
            else:
                # self.opp_possession_counter[agentID] += 1
                if (np.array(self.agent_possession_opp) == 'R').any():
                    prev_poss = (np.array(self.agent_possession_opp) == 'R').argmax()
                    if not self.agent_possession_opp[agentID] == 'R':
                        self.opp_passer[prev_poss] += 1 # sets passer flag to whoever passed
                        reward += self.pass_reward
                        team_reward += self.pass_reward
                        # print('opp pass reward received ')

                if (np.array(self.agent_possession_team) == 'L').any():
                    enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
                    self.team_lost_possession[enemy_possessor] -= 10.0
                    self.opp_lost_possession[agentID] += 10.0

                    # print('teammates lost possession ')

                self.agent_possession_team = ['N'] * self.num_TA
                self.agent_possession_opp = ['N'] * self.num_OA
                self.agent_possession_opp[agentID] = 'R'
                if possession_side != 'R':
                    possession_side = 'R'
                    #reward+=1
                    #team_reward+=1

        ####################### reduce distance to ball - using delta  ##################
        # if self.feat_lvl == 'high':
        #     r,_,_ = self.distance_to_ball(team_obs[agentID])
        #     r_prev,_,_ = self.distance_to_ball(team_obs_previous[agentID])
        #     reward += (r_prev - r) # if [prev > r] ---> positive; if [r > prev] ----> negative
        # elif self.feat_lvl == 'low':
        #     prox_cur = self.ball_proximity(team_obs[agentID])
        #     prox_prev = self.ball_proximity(team_obs_previous[agentID])
        #     reward   += (3)*(prox_cur - prox_prev) # if cur > prev --> +   
        #     team_reward +=(3)*(prox_cur - prox_prev)
                
        ####################### Rewards the closest player to ball for advancing toward ball ############
        if self.feat_lvl == 'low':
           prox_cur,_ = self.closest_player_to_ball(team_obs, num_ag)
           prox_prev,closest_agent = self.closest_player_to_ball(team_obs_previous, num_ag)
           if agentID == closest_agent:
               team_reward += (prox_cur - prox_prev)*10.0
               reward+= (prox_cur-prox_prev)*10.0
            
        ##################################################################################
            
        ####################### reduce ball distance to goal - For ball possessor and all agents on non controlling team  ##################
        r,_,_ = self.ball_distance_to_goal(team_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        r_prev,_,_ = self.ball_distance_to_goal(team_obs_previous[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        if ((self.team_base == base) and possession_side =='L'):
            team_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
            if agentID == team_possessor:
                reward += (10)*(r_prev - r)
                team_reward += (10)*(r_prev - r)
        elif  ((self.team_base != base) and possession_side == 'R'):
            team_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
            if agentID == team_possessor:
                reward += (10)*(r_prev - r)
                team_reward += (10)*(r_prev - r)
        else:
            reward += (10)*(r_prev - r)
            team_reward += (10)*(r_prev - r)

        

        
        ################## Offensive Behavior #######################
        # [Offense behavior]  agents will be rewarded based on maximizing their open angle to opponents goal
        if ((self.team_base == base) and possession_side =='L') or ((self.team_base != base) and possession_side == 'R'): # someone on team has ball
            b,_,_ =self.ball_distance_to_goal(team_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
            if b < 1.0 : # Ball is in scoring range
                if (self.apprx_to_goal(team_obs[agentID]) > .3) and (self.apprx_to_goal(team_obs[agentID]) < .85):
                    a = self.unnormalize(team_obs[agentID][self.open_goal])
                    a_prev = self.unnormalize(team_obs_previous[agentID][self.open_goal])
                    reward += (a-a_prev)*3
                    team_reward += (a-a_prev)*3
                    #print("offense behavior: goal angle open ",(a-a_prev)*3.0)


        # [Offense behavior]  agents will be rewarded based on maximizing their open angle to the ball (to receive pass)
        if ((self.team_base == base) and possession_side =='L') or ((self.team_base != base) and possession_side == 'R'): # someone on team has ball
            if (self.team_base != base):
                
                team_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
                #print("possessor is base right agent",team_possessor)

            else:
                team_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
                #print("possessor is base left agent",team_possessor)

            unif_nums = np.array([self.unnormalize_unif(val) for val in team_obs[team_possessor][self.team_unif_beg:self.team_unif_end]])
            unif_nums_prev = np.array([self.unnormalize_unif(val) for val in team_obs_previous[team_possessor][self.team_unif_beg:self.team_unif_end]])
            if agentID != team_possessor:
                if not (-100 in unif_nums) and not (-100 in unif_nums_prev):
                    angle_delta = self.unnormalize(team_obs[team_possessor][self.team_pass_angle_beg + np.argwhere(unif_nums == (agentID+1))[0][0]]) - self.unnormalize(team_obs_previous[team_possessor][self.team_pass_angle_beg+np.argwhere(unif_nums_prev == (agentID+1))[0][0]])
                    reward += angle_delta*3.0
                    team_reward += angle_delta*3.0
                    #print("offense behavior: pass angle open ",angle_delta*3.0)

                    

        ################## Defensive behavior ######################

        # [Defensive behavior]  agents will be rewarded based on minimizing the opponents open angle to our goal
        if ((self.team_base != base) and possession_side == 'L'): # someone on team has ball
            enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
            #print("possessor is base left agent",enemy_possessor)
            agent_inds = np.where([self.apprx_to_goal(opp_obs[i]) > .3 for i in range(self.num_TA)])[0] # find who is in range

            b,_,_ =self.ball_distance_to_goal(opp_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
            if b < 1.0 : # Ball is in scoring range
                if np.array([self.apprx_to_goal(opp_obs[i]) > .3 for i in range(self.num_TA)]).any(): # if anyone is in range on enemy team
                    sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[i][self.open_goal]) - self.unnormalize(opp_obs[i][self.open_goal])) for i in agent_inds]) # penalize based on the open angles of the people in range
                    reward += sum_angle_delta*3.0/self.num_TA
                    team_reward += sum_angle_delta*3.0/self.num_TA
                    angle_delta_possessor = self.unnormalize(opp_obs_previous[enemy_possessor][self.open_goal]) - self.unnormalize(opp_obs[enemy_possessor][self.open_goal])# penalize based on the open angles of the possessor
                    reward += angle_delta_possessor*3.0
                    team_reward += angle_delta_possessor*3.0  
                    #print("defensive behavior: block open angle to goal",agent_inds)
                    #print("areward for blocking goal: ",angle_delta_possessor*3.0)

        elif ((self.team_base == base) and possession_side =='R'): 
            enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
            #print("possessor is base right agent",enemy_possessor)
            agent_inds = np.where([self.apprx_to_goal(opp_obs[i]) > .3 for i in range(self.num_TA)])[0] # find who is in range

            b,_,_ =self.ball_distance_to_goal(opp_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
            if b < 1.0 : # Ball is in scoring range
                if np.array([self.apprx_to_goal(opp_obs[i]) > .3 for i in range(self.num_TA)]).any(): # if anyone is in range on enemy team
                    sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[i][self.open_goal]) - self.unnormalize(opp_obs[i][self.open_goal])) for i in agent_inds]) # penalize based on the open angles of the people in range
                    reward += sum_angle_delta*3.0/self.num_TA
                    team_reward += sum_angle_delta*3.0/self.num_TA
                    angle_delta_possessor = self.unnormalize(opp_obs_previous[enemy_possessor][self.open_goal]) - self.unnormalize(opp_obs[enemy_possessor][self.open_goal])# penalize based on the open angles of the possessor
                    reward += angle_delta_possessor*3.0
                    team_reward += angle_delta_possessor*3.0  
                    #print("defensive behavior: block open angle to goal",agent_inds)
                    #print("areward for blocking goal: ",angle_delta_possessor*3.0)




        # [Defensive behavior]  agents will be rewarded based on minimizing the ball open angle to other opponents (to block passes )
        if ((self.team_base != base) and possession_side == 'L'): # someone on team has ball
            enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
            sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[enemy_possessor][self.team_pass_angle_beg+i]) - self.unnormalize(opp_obs[enemy_possessor][self.team_pass_angle_beg+i])) for i in range(self.num_TA-1)]) # penalize based on the open angles of the people in range
            reward += sum_angle_delta*6.0/float(self.num_TA)
            team_reward += sum_angle_delta*6.0/float(self.num_TA)
            #print("defensive behavior: block open passes",enemy_possessor,"has ball")
            #print("reward for blocking",sum_angle_delta*3.0/self.num_TA)

        elif ((self.team_base == base) and possession_side =='R'):
            enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
            sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[enemy_possessor][self.team_pass_angle_beg+i]) - self.unnormalize(opp_obs[enemy_possessor][self.team_pass_angle_beg+i])) for i in range(self.num_TA-1)]) # penalize based on the open angles of the people in range
            reward += sum_angle_delta*6.0/float(self.num_TA)
            team_reward += sum_angle_delta*6.0/float(self.num_TA)
            #print("defensive behavior: block open passes",enemy_possessor,"has ball")
            #print("reward for blocking",sum_angle_delta*6.0/float(self.num_TA))



        ##################################################################################
        rew_percent = 1.0*max(0,(self.team_rew_anneal_ep - ep_num))/self.team_rew_anneal_ep
        return ((1.0 - rew_percent)*team_reward) + (reward * rew_percent)







    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self,port,feat_lvl, base, goalie, agent_ID,fpt,act_lvl):
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

        config_dir=get_config_path() 
        recorder_dir = 'log/'
        if self.team_base == base:
            self.team_envs[agent_ID].connectToServer(feat_lvl, config_dir=config_dir,
                                server_port=port, server_addr='localhost', team_name=base,
                                                    play_goalie=goalie,record_dir =recorder_dir)
        else:
            self.opp_team_envs[agent_ID].connectToServer(feat_lvl, config_dir=config_dir,
                                server_port=port, server_addr='localhost', team_name=base, 
                                                    play_goalie=goalie,record_dir =recorder_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Once all agents have been loaded,
        # wait for action command, take action, update: obs, reward, and world status
        ep_num = 0

        while(True):
            while(self.start):
                ep_num += 1
                j = 0 # j to maximum episode length
                # self.sync_at_status_team = np.zeros(self.num_TA)
                # self.sync_at_status_opp = np.zeros(self.num_OA)
                # self.sync_at_reward_team = np.zeros(self.num_TA)
                # self.sync_at_reward_opp = np.zeros(self.num_OA)

                if self.team_base == base:
                    self.team_obs_previous[agent_ID,:-3] = self.team_envs[agent_ID].getState() # Get initial state
                    self.team_obs[agent_ID,:-3] = self.team_envs[agent_ID].getState() # Get initial state
                    self.team_obs[agent_ID,-3] = 0
                    self.team_obs[agent_ID,-2] = 0
                    self.team_obs[agent_ID,-1] = 0

                    self.team_obs_previous[agent_ID,-3] = 0
                    self.team_obs_previous[agent_ID,-2] = 0
                    self.team_obs_previous[agent_ID,-1] = 0



                else:
                    self.opp_team_obs_previous[agent_ID,:-3] = self.opp_team_envs[agent_ID].getState() # Get initial state
                    self.opp_team_obs[agent_ID,:-3] = self.opp_team_envs[agent_ID].getState() # Get initial state
                    self.opp_team_obs[agent_ID,-3] = 0
                    self.opp_team_obs[agent_ID,-2] = 0
                    self.opp_team_obs[agent_ID,-1] = 0

                    self.opp_team_obs_previous[agent_ID,-3] = 0
                    self.opp_team_obs_previous[agent_ID,-2] = 0
                    self.opp_team_obs_previous[agent_ID,-1] = 0


                self.been_kicked_team = False
                self.been_kicked_opp = False
                while j < fpt:
                    #time.sleep(0.05)
                    # self.wait_for_queue = True

                    # if self.team_base == base:
                    #     self.sync_after_queue_team[agent_ID] += 1
                    # else:
                    #     self.sync_after_queue_opp[agent_ID] += 1

                    # while self.wait_for_queue:
                    #     time.sleep(self.sleep_timer)
                    #print('Done queueing actions')
                    self.sync_after_queue.wait()
                    self.team_obs[agent_ID,-1] = (j*1.0)/fpt
                    self.opp_team_obs[agent_ID,-1] = (j*1.0)/fpt

                    
                    
                    # take the action
                    if self.team_base == base:
                        # take the action
                        if act_lvl == 'high':
                            self.team_envs[agent_ID].act(self.action_list[self.team_actions[agent_ID]]) # take the action
                            self.sync_at_status_team[agent_ID] += 1
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
                                self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,0,base),self.get_valid_scaled_param(agent_ID,1,base))
                                #print(self.action_list[a],self.action_params[agent_ID][0],self.action_params[agent_ID][1])
                            elif a == 1:
                                #print(self.action_list[a],self.action_params[agent_ID][2])
                                self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,2,base))                       
                            elif a ==2:
                                #print(self.action_list[a],self.action_params[agent_ID][4],self.action_params[agent_ID][5])
                                self.team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,3,base),self.get_valid_scaled_param(agent_ID,4,base))

                            # self.sync_at_status_team[agent_ID] += 1
                    else:
                        # take the action
                        if act_lvl == 'high':
                            self.opp_team_envs[agent_ID].act(self.action_list[self.opp_actions[agent_ID]]) # take the action
                            self.sync_at_status_opp[agent_ID] += 1
                        elif act_lvl == 'low':
                            # use params for low level actions
                            # scale action params
                            a = self.opp_actions[agent_ID]
                            
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
                                self.opp_team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,0,base),self.get_valid_scaled_param(agent_ID,1,base))
                                #print(self.action_list[a],self.action_params[agent_ID][0],self.action_params[agent_ID][1])
                            elif a == 1:
                                #print(self.action_list[a],self.action_params[agent_ID][2])
                                self.opp_team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,2,base))                       
                            elif a == 2:
                                #print(self.action_list[a],self.action_params[agent_ID][4],self.action_params[agent_ID][5])
                                self.opp_team_envs[agent_ID].act(self.action_list[a],self.get_valid_scaled_param(agent_ID,3,base),self.get_valid_scaled_param(agent_ID,4,base))

                            # self.sync_at_status_opp[agent_ID] += 1

                    # while (self.sync_at_status_team.sum() + self.sync_at_status_opp.sum()) % (self.num_TA + self.num_OA) != 0:
                    #     time.sleep(self.sleep_timer)
                    self.sync_at_status.wait()
                    
                    if self.team_base == base:
                        self.team_obs_previous[agent_ID] = self.team_obs[agent_ID]
                        self.world_status = self.team_envs[agent_ID].step() # update world
                        self.team_obs[agent_ID,:-2] = self.team_envs[agent_ID].getState() # update obs after all agents have acted
                        # self.sync_at_reward_team[agent_ID] += 1
                    else:
                        self.opp_team_obs_previous[agent_ID] = self.opp_team_obs[agent_ID]
                        self.world_status = self.opp_team_envs[agent_ID].step() # update world
                        self.opp_team_obs[agent_ID,:-2] = self.opp_team_envs[agent_ID].getState() # update obs after all agents have acted
                        # self.sync_at_reward_opp[agent_ID] += 1
                    
                    # while (self.sync_at_reward_team.sum() + self.sync_at_reward_opp.sum()) % (self.num_TA + self.num_OA) != 0:
                    #     time.sleep(self.sleep_timer)
                    self.sync_at_reward.wait()

                    if self.world_status == hfo.IN_GAME:
                        self.d = 0
                    else:
                        self.d = 1

                    if self.team_base == base:
                        self.team_rewards[agent_ID] = self.getReward(
                            self.team_envs[agent_ID].statusToString(self.world_status),agent_ID,base,ep_num) # update reward
                        # self.sync_before_step_team[agent_ID] += 1
                    else:
                        self.opp_rewards[agent_ID] = self.getReward(
                            self.opp_team_envs[agent_ID].statusToString(self.world_status),agent_ID,base,ep_num) # update reward
                        # self.sync_before_step_opp[agent_ID] += 1

                    j+=1

                    # while (self.sync_before_step_team.sum() + self.sync_before_step_opp.sum()) % (self.num_TA + self.num_OA) != 0:
                    #     time.sleep(self.sleep_timer)
                    self.sync_before_step.wait()

                    # self.wait_for_connect_vals = False

                    # Break if episode done
                    if self.d == True:
                        break


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # found from https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    def _start_hfo_server(self, frames_per_trial=100,
                              untouched_time=100, offense_agents=1,
                              defense_agents=0, offense_npcs=0,
                              defense_npcs=0, sync_mode=True, port=6000,
                              offense_on_ball=0, fullstate=False, seed=123,
                              ball_x_min=-0.8, ball_x_max=0.8,
                              ball_y_min=-0.8, ball_y_max=0.8,
                              verbose=False, rcss_log_game=False,
                              log_dir="log",
                              hfo_log_game=True,
                              agents_x_min=0.0, agents_x_max=0.0,
                              agents_y_min=0.0, agents_y_max=0.0,
                              change_every_x=1, change_agents_x=0.1,
                              change_agents_y=0.1, change_balls_x=0.1,
                              change_balls_y=0.1, control_rand_init=False,record=True,
                              defense_team_bin='base', offense_team_bin='helios16', deterministic=True):
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
            ball_x_[min/max]: Initialize the ball this far downfield: [-1,1]
            verbose: Verbose server messages.
            log_game: Enable game logging. Logs can be used for replay + visualization.
            log_dir: Directory to place game logs (*.rcg).
            """
            self.server_port = port
            cmd = self.hfo_path + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                  " --log-dir %s --message-size 256"\
                  % (frames_per_trial, untouched_time, offense_agents,
                     defense_agents, offense_npcs, defense_npcs, port,
                     offense_on_ball, seed, ball_x_min, ball_x_max,
                     ball_y_min, ball_y_max, log_dir)
            #Adds the binaries when offense and defense npcs are in play, must be changed to add agent vs binary npc
            if offense_npcs > 0:   cmd += " --offense-team %s" \
                % (offense_team_bin)
            if defense_npcs > 0:   cmd += " --defense-team %s" \
                % (defense_team_bin)
            if not sync_mode:      cmd += " --no-sync"
            if fullstate:          cmd += " --fullstate"
            if deterministic:      cmd += " --deterministic"
            if verbose:            cmd += " --verbose"
            if not rcss_log_game:  cmd += " --no-logging"
            if hfo_log_game:       cmd += " --hfo-logging"
            if record:             cmd += " --record"
            if control_rand_init:
                cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                        " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                        " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                        % (agents_x_min, agents_x_max, agents_y_min, agents_y_max,
                            change_every_x, change_agents_x, change_agents_y,
                            change_balls_x, change_balls_y)

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
        cmd = get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)
    
    # Only will work for 2 v 2
    def test_obs_validity(self, base):
        observations = None
        exit_check = False
        if base == self.team_base:
            observations = self.team_obs
            print('side chosen', base)
        else:
            observations = self.opp_team_obs
            print('side chosen', base)
        
        if observations[0][0:2].any() == -1 or observations[1][0:2].any() == -1:
            print('agent:0/1, pos/velocity invalid')
            exit_check = True
        
        if observations[0][2:5].any() == -2 or observations[1][2:5].any() == -2:
            print('agent:0/1, velocity ang/mag invalid')
            exit_check = True
        
        if observations[0][13:46].any() == -2 or observations[1][13:46].any() == -2:
            print('agent:0/1, landmark invalid')
            exit_check = True
        
        if observations[0][46:50].any() == -2 or observations[1][46:50].any() == -2:
            print('agent:0/1, OOB invalid')
            exit_check = True
        
        if observations[0][50] == -1 or observations[1][50] == -1:
            print('agent:0/1, Ball pos invalid')
            exit_check = True

        if observations[0][54] == -1 or observations[1][54] == -1:
            print('agent:0/1, Ball velocity invalid')
            exit_check = True
        
        if observations[0][59] == -2 or observations[1][59] == -2:
            print('agent:0/1, teammate agents not detected for open goal invalid')
            exit_check = True
        
        if observations[0][60:62].any() == -2 or observations[1][60:62].any() == -2:
            print('agent:0/1, opponent agents not detected for open goal invalid')
            exit_check = True
        
        if observations[0][62] == -2 or observations[1][62] == -2:
            print('agent:0/1, open angle to teammates invalid')
            exit_check = True
        
        if observations[0][63:71].any() == -2 or observations[1][63:71].any() == -2:
            print('agent:0/1, teammate player features invalid')
            exit_check = True
        
        if observations[0][71:71+(2*8)].any() == -2 or observations[1][71:71+(2*8)].any() == -2:
            print('agent:0/1, opponent player features invalid')
            exit_check = True
        
        if observations[0][87] == -1 or observations[1][87] == -1:
            print('agent:0/1, teammate uniform invalid')
            exit_check = True
        
        if observations[0][88:90].any() == -1 or observations[1][88:90].any() == -1:
            print('agent:0/1, opponent uniform invalid')
            exit_check = True
        
        # if(self.test == 10):
        #     print('Self x, y', observations[0][90:92], observations[1][90:92])
        #     print('Teamates x,y', observations[0][92:94], observations[1][92:94])
        #     print('Opponents x,y', observations[0][94:98], observations[1][94:98])
        #     print('Ball x,y', observations[0][98:100], observations[1][98:100])
        #     exit(0)
        # self.test+=1
        
        if exit_check:
            print('Exiting program')
            exit(0)

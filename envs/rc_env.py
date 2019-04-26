import random
import numpy as np
import time
import threading
import pandas as pd
import math
from . base_env import Base_Env
from HFO.hfo import hfo
from HFO import get_config_path, get_hfo_path, get_viewer_path
import os, subprocess, time, signal
from utils import misc as misc
from algorithms.maddpg import MADDPG
from torch.autograd import Variable
import torch


possession_side = 'N'

class rc_env(Base_Env):
    """rc_env() extends the HFO environment to allow for centralized execution.
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
    # def __init__(self, num_TNPC=0,num_TA = 1,num_OA = 0,num_ONPC = 1,base = 'base_left',
    #              goalie = False, num_trials = 10000,fpt = 100,feat_lvl = 'high',
    #              act_lvl = 'low',untouched_time = 100, sync_mode = True, port = 6000,
    #              offense_on_ball=0, fullstate = False, seed = 123,
    #              ball_x_min = -0.8, ball_x_max = 0.8, ball_y_min = -0.8, ball_y_max = 0.8,
    #              verbose = False, rcss_log_game=False, hfo_log_game=False, log_dir="log",team_rew_anneal_ep=1000,
    #              agents_x_min=-0.8, agents_x_max=0.8, agents_y_min=-0.8, agents_y_max=0.8,
    #              change_every_x=5, change_agents_x=0.1, change_agents_y=0.1, change_balls_x=0.1,
    #              change_balls_y=0.1, control_rand_init=False,record=False,record_server=False,
    #              defense_team_bin='helios15', offense_team_bin='helios16', run_server=False, deterministic=True):

    def __init__(self, config):
        self.config = config

        self.pass_reward = 0.0
        self.agent_possession_team = ['N'] * config.num_left
        self.team_passer = [0]*config.num_left
        self.opp_passer = [0]*config.num_left
        self.team_lost_possession = [0]*config.num_left
        self.opp_lost_possession = [0]*config.num_left
        self.agent_possession_opp = ['N'] * config.num_right
        self.team_possession_counter = [0] * config.num_left
        self.opp_possession_counter = [0] * config.num_right
        self.team_kickable = [0] * config.num_right
        self.opp_kickable = [0] * config.num_right
        self.offense_on_ball = config.offense_ball

        self.untouched = config.untouched

        self.goalie = config.goalie
        self.team_rew_anneal_ep = config.reward_anneal
        self.port = config.port
        self.hfo_path = get_hfo_path()
        self.seed = np.random.randint(1000)

        self.viewer = None

        # params for low level actions
        num_action_params = 5 # 2 for dash and kick 1 for turn and tackle
        self.team_action_params = np.asarray([[0.0]*num_action_params for i in range(config.num_left)])
        self.opp_action_params = np.asarray([[0.0]*num_action_params for i in range(config.num_right)])
        if config.al == 'low':
            #                   pow,deg   deg       deg         pow,deg    
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo.DASH, hfo.TURN, hfo.KICK]

            self.kick_actions = [hfo.KICK] # actions that require the ball to be kickable
    
        elif config.al == 'high':
            self.action_list = [hfo.DRIBBLE, hfo.SHOOT, hfo.REORIENT, hfo.GO_TO_BALL, hfo.MOVE]
            self.kick_actions = [hfo.DRIBBLE, hfo.SHOOT, hfo.PASS] # actions that require the ball to be kickable
        
        self.num_TA = config.num_left
        self.num_OA = config.num_right
        self.num_ONPC = config.num_r_bot
        self.num_TNPC = config.num_l_bot

        self.base = 'base_left'
        self.fpt = config.ep_length
        self.been_kicked_team = False
        self.been_kicked_opp = False
        self.act_lvl = config.al
        self.feat_lvl = config.fl
        self.team_possession = False
        self.opp_possession = False
        if config.fl == 'low':
            #For new obs reorganization without vailds, changed hfo obs from 59 to 56
            self.team_num_features = 56 + 13*(config.num_left-1) + 12*config.num_right + 4 + 1 + 2 + 1  + 8
            self.opp_num_features = 56 + 13*(config.num_right-1) + 12*config.num_left + 4 + 1 + 2 + 1 + 8
        elif config.fl == 'high':
            self.team_num_features = (6*config.num_left) + (3*config.num_right) + (3*num_ONPC) + 6
            self.opp_num_features = (6*config.num_right) + (3*config.num_left) + (3*num_ONPC) + 6
        elif config.fl == 'simple':
            # 16 - land_feats + 12 - basic feats + 6 per (team/opp)
            self.team_num_features = 28 + (6 * ((config.num_left-1) + config.num_right)) + 8
            self.opp_num_features = 28 + (6 * (config.num_left + (config.num_right-1))) + 8

        self.acs_dim = config.ac_dim
        # Feature indexes by name
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
        # Create env for each teammate
        self.team_envs = [hfo.HFOEnvironment() for i in range(config.num_left)]
        self.opp_team_envs = [hfo.HFOEnvironment() for i in range(config.num_right)]

        # flag that says when the episode is done
        self.d = 0

        # flag to wait for all the agents to load
        self.start = False

        self.sync_after_queue = threading.Barrier(config.num_left+config.num_right+1)
        self.sync_before_step = threading.Barrier(config.num_left+config.num_right+1)
        self.sync_at_status = threading.Barrier(config.num_left+config.num_right)
        self.sync_at_reward = threading.Barrier(config.num_left+config.num_right)

        # Initialization of mutable lists to be passsed to threads
        # action each team mate is supposed to take when its time to act
        self.team_actions = np.array([2]*config.num_left)
        # observation space for all team mate agents
        self.team_obs = np.empty([config.num_left,self.team_num_features],dtype=float)
        # previous state for all team agents
        self.team_obs_previous = np.empty([config.num_left,self.team_num_features],dtype=float)
        self.team_actions_OH = np.empty([config.num_left,8],dtype=float)
        self.opp_actions_OH = np.empty([config.num_left,8],dtype=float)

        # reward for each agent
        self.team_rewards = np.zeros(config.num_left)

        self.opp_actions = np.array([2]*config.num_right)
        # observation space for all team mate agents
        self.opp_team_obs = np.empty([config.num_right,self.opp_num_features],dtype=float)
        # previous state for all team agents
        self.opp_team_obs_previous = np.empty([config.num_right,self.opp_num_features],dtype=float)
        # reward for each agent
        self.opp_rewards = np.zeros(config.num_right)

        # keeps track of world state
        self.world_status = 0
        
        self.team_base = self.base
        self.opp_base = ''

        # Create thread for each opponent (set one to goalie)
        if self.base == 'base_left':
            self.opp_base = 'base_right'
        elif self.base == 'base_right':
            self.opp_base = 'base_left'

        
    def launch(self):
        # self._start_hfo_server(frames_per_trial = fpt, untouched_time = untouched_time,
        #                             offense_agents = num_TA, defense_agents = num_OA,
        #                             offense_npcs = num_TNPC, defense_npcs = num_ONPC,
        #                             sync_mode = sync_mode, port = port,
        #                             offense_on_ball = offense_on_ball,
        #                             fullstate = fullstate, seed = seed,
        #                             ball_x_min = ball_x_min, ball_x_max = ball_x_max,
        #                             ball_y_min= ball_y_min, ball_y_max= ball_y_max,
        #                             verbose = verbose, rcss_log_game = rcss_log_game, 
        #                             hfo_log_game=hfo_log_game, log_dir = log_dir,
        #                             agents_x_min=agents_x_min, agents_x_max=agents_x_max,
        #                             agents_y_min=agents_y_min, agents_y_max=agents_y_max,
        #                             change_every_x=change_every_x, change_agents_x=change_agents_x,
        #                             change_agents_y=change_agents_y, change_balls_x=change_balls_x,
        #                             change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=record,record_server=record_server,
        #                             defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, deterministic=deterministic)
        self._start_hfo_server()
        
        time.sleep(3)
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


    def Step(self, team_actions, opp_actions, team_params=[], opp_params=[],team_actions_OH = [],opp_actions_OH = []):
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
        for i in range(self.num_TA):
            self.team_actions_OH[i] = misc.zero_params(team_actions_OH[i].reshape(-1))
            self.opp_actions_OH[i] = misc.zero_params(opp_actions_OH[i].reshape(-1))
        [self.Queue_action(i,self.team_base,team_actions[i],team_params) for i in range(len(team_actions))]
        # Queue actions for opposing team
        [self.Queue_action(j,self.opp_base,opp_actions[j],opp_params) for j in range(len(opp_actions))]

        self.sync_after_queue.wait()

        self.sync_before_step.wait()

        self.team_rewards = [rew + self.pass_reward if passer else rew for rew,passer in zip(self.team_rewards,self.team_passer)]
        self.opp_rewwards =[rew + self.pass_reward if passer else rew for rew,passer in zip(self.opp_rewards,self.opp_passer)]
        

        self.team_rewards = np.add( self.team_rewards, self.team_lost_possession)
        self.opp_rewards = np.add(self.opp_rewards, self.opp_lost_possession)


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

    def get_kickable_status(self,agentID,env):
        ball_kickable = False
        ball_kickable = env.isKickable()
        #print("no implementation")
        return ball_kickable
    
    def get_agent_possession_status(self,agentID,base):
        if self.team_base == base:
            if self.agent_possession_team[agentID] == 'L':
                self.team_possession_counter[agentID] += 1
            
            return self.team_possession_counter[agentID]
        else:
            if self.agent_possession_opp[agentID] == 'R':
                self.opp_possession_counter[agentID] += 1
        #    print("no implementation")
            return self.opp_possession_counter[agentID]  



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
        print("no implementation")
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
        #print("no implementation")
        return 0.0
        
        
    def ball_distance_to_goal(self,obs):
        goal_center_x = 1.0
        goal_center_y = 0.0
        relative_x = obs[self.ball_x] - goal_center_x
        relative_y = obs[self.ball_y] - goal_center_y
        ball_distance_to_goal = math.sqrt(relative_x**2 + relative_y**2)
        return ball_distance_to_goal
    
 
    def distance_to_ball(self, obs):
        relative_x = obs[self.x]-obs[self.ball_x]
        relative_y = obs[self.y]-obs[self.ball_y]
        ball_distance = math.sqrt(relative_x**2+relative_y**2)
        
        return ball_distance
    
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
        goal_points = 30.0
        #---------------------------
        global possession_side
        if d:
            if self.team_base == base:
            # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----
                if s==1:
                    reward+= goal_points
                elif s==2:
                    reward+=-goal_points
                elif s==3:
                    reward+=-0.5
                elif s==6:
                    reward+= +goal_points/5.0
                elif s==7:
                    reward+= -goal_points/4.0

                return reward
            else:
                if s==1:
                    reward+=-goal_points
                elif s==2:
                    reward+=goal_points
                elif s==3:
                    reward+=-0.5
                elif s==6:
                    reward+= -goal_points/4.0
                elif s==7:
                    reward+= goal_points/5.0

        return reward
       
    def distance(self,x1,x2,y1,y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def distances(self,agentID,side):
        if side == 'left':
            team_obs = self.team_obs
            opp_obs =  self.opp_team_obs
        elif side =='right':
            team_obs = self.opp_team_obs
            opp_obs = self.team_obs
        else:
            print("Error: Please return a side: ('left', 'right') for side parameter")

        distances_team = []
        distances_opp = []
        for i in range(len(team_obs)):
            distances_team.append(self.distance(team_obs[agentID][self.x],team_obs[i][self.x], team_obs[agentID][self.y],team_obs[i][self.y]))
            distances_opp.append(self.distance(team_obs[agentID][self.x], -opp_obs[i][self.x], team_obs[agentID][self.y], -opp_obs[i][self.y]))
        return np.argsort(distances_team), np.argsort(distances_opp)


    def unnormalize(self,val):
        return (val +1.0)/2.0
    
    # Engineered Reward Function
    #   To-do: Add the ability of team agents to know if opponent npcs hold ball posession. For now, having npc opponent disables first kick award
    def getReward(self,s,agentID,base,ep_num):
        reward=0.0
        team_reward = 0.0
        goal_points = 20.0
        #---------------------------
        global possession_side
        if self.d:
            if self.team_base == base:
            # ------- If done with episode, don't calculate other rewards (reset of positioning messes with deltas) ----
                if s=='Goal_By_Left' and self.agent_possession_team[agentID] == 'L':
                    reward+= goal_points
                elif s=='Goal_By_Left':
                    reward+= goal_points # teammates get 10% of points
                elif s=='Goal_By_Right':
                    reward+=-goal_points
                elif s=='OutOfBounds' and self.agent_possession_team[agentID] == 'L':
                    reward+=-0.5
                elif s=='CapturedByLeftGoalie':
                    reward+=goal_points/5.0
                elif s=='CapturedByRightGoalie':
                    reward+= 0 #-goal_points/4.0

                possession_side = 'N' # at the end of each episode we set this to none
                self.agent_possession_team = ['N'] * self.num_TA
                return reward
            else:
                if s=='Goal_By_Right' and self.agent_possession_opp[agentID] == 'R':
                    reward+=goal_points
                elif s=='Goal_By_Right':
                    reward+=goal_points
                elif s=='Goal_By_Left':
                    reward+=-goal_points
                elif s=='OutOfBounds' and self.agent_possession_opp[agentID] == 'R':
                    reward+=-0.5
                elif s=='CapturedByRightGoalie':
                    reward+=goal_points/5.0
                elif s=='CapturedByLeftGoalie':
                    reward+= 0 #-goal_points/4.0

                possession_side = 'N'
                self.agent_possession_opp = ['N'] * self.num_OA
                return reward
        

        
        if self.team_base == base:
            team_actions = self.team_actions
            team_obs = self.team_obs
            team_obs_previous = self.team_obs_previous
            opp_obs = self.opp_team_obs
            opp_obs_previous = self.opp_team_obs_previous
            num_ag = self.num_TA
            env = self.team_envs[agentID]
            kickable = self.team_kickable[agentID]
            self.team_kickable[agentID] = self.get_kickable_status(agentID,env)
        else:
            team_actions = self.opp_actions
            team_obs = self.opp_team_obs
            team_obs_previous = self.opp_team_obs_previous
            opp_obs = self.team_obs
            opp_obs_previous = self.team_obs_previous
            num_ag = self.num_OA
            env = self.opp_team_envs[agentID]
            kickable = self.opp_kickable[agentID]
            self.opp_kickable[agentID] = self.get_kickable_status(agentID,env)# update kickable status (it refers to previous timestep, e.g., it WAS kickable )

        if team_obs[agentID][self.stamina] < 0.0 : # LOW STAMINA
            reward -= 0.003
            team_reward -= 0.003
            # print ('low stamina')
        


        ############ Kicked Ball #################
        
        if self.action_list[team_actions[agentID]] in self.kick_actions and kickable:            
            if self.num_OA > 0:
                if (np.array(self.agent_possession_team) == 'N').all() and (np.array(self.agent_possession_opp) == 'N').all():
                     #print("First Kick")
                    reward += 1.5
                    team_reward +=1.5
                # set initial ball position after kick
                    if self.team_base == base:
                        self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
                        self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
                    else:
                        self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
                        self.BR_ball_pos_y = team_obs[agentID][self.ball_y]
                        

        #     # track ball delta in between kicks
            if self.team_base == base:
                self.BL_ball_pos_x = team_obs[agentID][self.ball_x]
                self.BL_ball_pos_y = team_obs[agentID][self.ball_y]
            else:
                self.BR_ball_pos_x = team_obs[agentID][self.ball_x]
                self.BR_ball_pos_y = team_obs[agentID][self.ball_y]

            new_x = team_obs[agentID][self.ball_x]
            new_y = team_obs[agentID][self.ball_y]
            
            if self.team_base == base:
                ball_delta = math.sqrt((self.BL_ball_pos_x-new_x)**2+ (self.BL_ball_pos_y-new_y)**2)
                self.BL_ball_pos_x = new_x
                self.BL_ball_pos_y = new_y
            else:
                ball_delta = math.sqrt((self.BR_ball_pos_x-new_x)**2+ (self.BR_ball_pos_y-new_y)**2)
                self.BR_ball_pos_x = new_x
                self.BR_ball_pos_y = new_y
            
            self.pass_reward = ball_delta * 5.0

        #     ######## Pass Receiver Reward #########
            if self.team_base == base:
                if (np.array(self.agent_possession_team) == 'L').any():
                    prev_poss = (np.array(self.agent_possession_team) == 'L').argmax()
                    if not self.agent_possession_team[agentID] == 'L':
                        self.team_passer[prev_poss] += 1 # sets passer flag to whoever passed
                        # Passer reward is added in step function after all agents have been checked
                       
                        reward += self.pass_reward
                        team_reward += self.pass_reward
                        #print("received a pass worth:",self.pass_reward)
        #               #print('team pass reward received ')
        #         #Remove this check when npc ball posession can be measured
                if self.num_OA > 0:
                    if (np.array(self.agent_possession_opp) == 'R').any():
                        enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
                        self.opp_lost_possession[enemy_possessor] -= 1.0
                        self.team_lost_possession[agentID] += 1.0
                        # print('BR lost possession')
                        self.pass_reward = 0

        #         ###### Change Possession Reward #######
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
        #                 # print('opp pass reward received ')

                if (np.array(self.agent_possession_team) == 'L').any():
                    enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
                    self.team_lost_possession[enemy_possessor] -= 1.0
                    self.opp_lost_possession[agentID] += 1.0
                    self.pass_reward = 0
      #             # print('BL lost possession ')

                self.agent_possession_team = ['N'] * self.num_TA
                self.agent_possession_opp = ['N'] * self.num_OA
                self.agent_possession_opp[agentID] = 'R'
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
        distance_cur,_ = self.closest_player_to_ball(team_obs, num_ag)
        distance_prev, closest_agent = self.closest_player_to_ball(team_obs_previous, num_ag)
        if agentID == closest_agent:
            delta = (distance_prev - distance_cur)*1.0
            #if delta > 0:    
            if True:
                team_reward += delta
                reward+= delta
            
        ##################################################################################
            
        ####################### reduce ball distance to goal ##################
        # base left kicks
        r = self.ball_distance_to_goal(team_obs[agentID]) 
        r_prev = self.ball_distance_to_goal(team_obs_previous[agentID]) 
        if ((self.team_base == base) and possession_side =='L'):
            team_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
            if agentID == team_possessor:
                delta = (2*self.num_TA)*(r_prev - r)
                if True:
                #if delta > 0:
                    reward += delta
                    team_reward += delta

        # base right kicks
        elif  ((self.team_base != base) and possession_side == 'R'):
            team_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
            if agentID == team_possessor:
                delta = (2*self.num_TA)*(r_prev - r)
                if True:
                #if delta > 0:
                    reward += delta
                    team_reward += delta
                    
        # non-possessor reward for ball delta toward goal
        else:
            delta = (0*self.num_TA)*(r_prev - r)
            if True:
            #if delta > 0:
                reward += delta
                team_reward += delta       

        # ################## Offensive Behavior #######################
        # # [Offense behavior]  agents will be rewarded based on maximizing their open angle to opponents goal ( only for non possessors )
        # if ((self.team_base == base) and possession_side =='L') or ((self.team_base != base) and possession_side == 'R'): # someone on team has ball
        #     b,_,_ =self.ball_distance_to_goal(team_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        #     if b < 1.5 : # Ball is in scoring range
        #         if (self.apprx_to_goal(team_obs[agentID]) > 0.0) and (self.apprx_to_goal(team_obs[agentID]) < .85):
        #             a = self.unnormalize(team_obs[agentID][self.open_goal])
        #             a_prev = self.unnormalize(team_obs_previous[agentID][self.open_goal])
        #             if (self.team_base != base):    
        #                 team_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
        #             else:
        #                 team_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
        #             if agentID != team_possessor:
        #                 reward += (a-a_prev)*2.0
        #                 team_reward += (a-a_prev)*2.0
        #             #print("offense behavior: goal angle open ",(a-a_prev)*3.0)


        # # [Offense behavior]  agents will be rewarded based on maximizing their open angle to the ball (to receive pass)
        # if self.num_TA > 1:
        #     if ((self.team_base == base) and possession_side =='L') or ((self.team_base != base) and possession_side == 'R'): # someone on team has ball
        #         if (self.team_base != base):
                    
        #             team_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
        #             #print("possessor is base right agent",team_possessor)

        #         else:
        #             team_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
        #             #print("possessor is base left agent",team_possessor)

        #         unif_nums = np.array([self.unnormalize_unif(val) for val in team_obs[team_possessor][self.team_unif_beg:self.team_unif_end]])
        #         unif_nums_prev = np.array([self.unnormalize_unif(val) for val in team_obs_previous[team_possessor][self.team_unif_beg:self.team_unif_end]])
        #         if agentID != team_possessor:
        #             if not (-100 in unif_nums) and not (-100 in unif_nums_prev):
        #                 angle_delta = self.unnormalize(team_obs[team_possessor][self.team_pass_angle_beg + np.argwhere(unif_nums == (agentID+1))[0][0]]) - self.unnormalize(team_obs_previous[team_possessor][self.team_pass_angle_beg+np.argwhere(unif_nums_prev == (agentID+1))[0][0]])
        #                 reward += angle_delta
        #                 team_reward += angle_delta
        #                 #print("offense behavior: pass angle open ",angle_delta*3.0)

                    

        # ################## Defensive behavior ######################

        # # [Defensive behavior]  agents will be rewarded based on minimizing the opponents open angle to our goal
        # if ((self.team_base != base) and possession_side == 'L'): # someone on team has ball
        #     enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
        #     #print("possessor is base left agent",enemy_possessor)
        #     agent_inds = np.where([self.apprx_to_goal(opp_obs[i]) > -0.75 for i in range(self.num_TA)])[0] # find who is in range

        #     b,_,_ =self.ball_distance_to_goal(opp_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8

        #     if b < 1.5 : # Ball is in scoring range
        #         if np.array([self.apprx_to_goal(opp_obs[i]) > -0.75 for i in range(self.num_TA)]).any(): # if anyone is in range on enemy team
        #             sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[i][self.open_goal]) - self.unnormalize(opp_obs[i][self.open_goal])) for i in agent_inds]) # penalize based on the open angles of the people in range
        #             reward += sum_angle_delta*2.0
        #             team_reward += sum_angle_delta*2.0
        #             angle_delta_possessor = self.unnormalize(opp_obs_previous[enemy_possessor][self.open_goal]) - self.unnormalize(opp_obs[enemy_possessor][self.open_goal])# penalize based on the open angles of the possessor
        #             reward += angle_delta_possessor*2.0
        #             team_reward += angle_delta_possessor*2.0
        #             #print("defensive behavior: block open angle to goal",agent_inds)
        #             #print("areward for blocking goal: ",angle_delta_possessor*3.0)

        # elif ((self.team_base == base) and possession_side =='R'): 
        #     enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
        #     #print("possessor is base right agent",enemy_possessor)
        #     agent_inds = np.where([self.apprx_to_goal(opp_obs[i]) > -0.75 for i in range(self.num_TA)])[0] # find who is in range

        #     b,_,_ =self.ball_distance_to_goal(opp_obs[agentID]) #r is maxed at 2sqrt(2)--> 2.8
        #     if b < 1.5 : # Ball is in scoring range
        #         if np.array([self.apprx_to_goal(opp_obs[i]) > -0.75 for i in range(self.num_TA)]).any(): # if anyone is in range on enemy team
        #             sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[i][self.open_goal]) - self.unnormalize(opp_obs[i][self.open_goal])) for i in agent_inds]) # penalize based on the open angles of the people in range
        #             reward += sum_angle_delta*2.0
        #             team_reward += sum_angle_delta*2.0
        #             angle_delta_possessor = self.unnormalize(opp_obs_previous[enemy_possessor][self.open_goal]) - self.unnormalize(opp_obs[enemy_possessor][self.open_goal])# penalize based on the open angles of the possessor
        #             reward += angle_delta_possessor*2.0
        #             team_reward += angle_delta_possessor*2.0
        #             #print("defensive behavior: block open angle to goal",agent_inds)
        #             #print("areward for blocking goal: ",angle_delta_possessor*3.0)




        # # [Defensive behavior]  agents will be rewarded based on minimizing the ball open angle to other opponents (to block passes )
        # if self.num_TA > 1:
                
        #     if ((self.team_base != base) and possession_side == 'L'): # someone on team has ball
        #         enemy_possessor = (np.array(self.agent_possession_team) == 'L').argmax()
        #         sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[enemy_possessor][self.team_pass_angle_beg+i]) - self.unnormalize(opp_obs[enemy_possessor][self.team_pass_angle_beg+i])) for i in range(self.num_TA-1)]) # penalize based on the open angles of the people in range
        #         reward += sum_angle_delta*0.3/float(self.num_TA)
        #         team_reward += sum_angle_delta*0.3/float(self.num_TA)
        #         #print("defensive behavior: block open passes",enemy_possessor,"has ball")
        #         #print("reward for blocking",sum_angle_delta*3.0/self.num_TA)

        #     elif ((self.team_base == base) and possession_side =='R'):
        #         enemy_possessor = (np.array(self.agent_possession_opp) == 'R').argmax()
        #         sum_angle_delta = np.sum([(self.unnormalize(opp_obs_previous[enemy_possessor][self.team_pass_angle_beg+i]) - self.unnormalize(opp_obs[enemy_possessor][self.team_pass_angle_beg+i])) for i in range(self.num_TA-1)]) # penalize based on the open angles of the people in range
        #         reward += sum_angle_delta*0.3/float(self.num_TA)
        #         team_reward += sum_angle_delta*0.3/float(self.num_TA)
        #         #print("defensive behavior: block open passes",enemy_possessor,"has ball")
        #         #print("reward for blocking",sum_angle_delta*6.0/float(self.num_TA))

        ##################################################################################
        rew_percent = 1.0*max(0,(self.team_rew_anneal_ep - ep_num))/self.team_rew_anneal_ep
        return ((1.0 - rew_percent)*team_reward) + (reward * rew_percent)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self,port,feat_lvl, base, goalie, agent_ID,fpt,act_lvl):
        """ Connect threaded agent to server
        Args:
            feat_lvl: Feature level to use. ('high', 'low', 'simple')
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
        elif feat_lvl == 'simple':
            feat_lvl = hfo.SIMPLE_LEVEL_FEATURE_SET

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

                if self.team_base == base:

                    self.team_obs_previous[agent_ID,:-8] = self.team_envs[agent_ID].getState() # Get initial state
                    self.team_obs[agent_ID,:-8] = self.team_envs[agent_ID].getState() # Get initial state
                    self.team_obs[agent_ID,-8:] = [0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0]
                    self.team_obs_previous[agent_ID,-8:] = [0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0]
                else:
                    self.opp_team_obs_previous[agent_ID,:-8] = self.opp_team_envs[agent_ID].getState() # Get initial state
                    self.opp_team_obs[agent_ID,:-8] = self.opp_team_envs[agent_ID].getState() # Get initial state
                    self.opp_team_obs[agent_ID,-8:] = [0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0]
                    self.opp_team_obs_previous[agent_ID,-8:] = [0.0,0.0,0.0, 0.0,0.0,0.0,0.0,0.0]


                self.been_kicked_team = False
                self.been_kicked_opp = False
                while j < fpt:

                    self.sync_after_queue.wait()

                    
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
                            elif a == 1:team_envs
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

                    self.sync_at_status.wait()
                    
                    if self.team_base == base:
                        self.team_obs_previous[agent_ID] = self.team_obs[agent_ID]
                        self.world_status = self.team_envs[agent_ID].step() # update world                        
                        self.team_obs[agent_ID,:-8] = self.team_envs[agent_ID].getState() # update obs after all agents have acted
                        self.team_obs[agent_ID,-8:] =  self.team_actions_OH[agent_ID]
                    else:
                        self.opp_team_obs_previous[agent_ID] = self.opp_team_obs[agent_ID]
                        self.world_status = self.opp_team_envs[agent_ID].step() # update world
                        self.opp_team_obs[agent_ID,:-8] = self.opp_team_envs[agent_ID].getState() # update obs after all agents have acted
                        self.opp_team_obs[agent_ID,-8:] =  self.opp_actions_OH[agent_ID]

                    self.sync_at_reward.wait()

                    if self.world_status == hfo.IN_GAME:
                        self.d = 0
                    else:
                        self.d = 1

                    if self.team_base == base:
                        self.team_rewards[agent_ID] = self.getReward(
                            self.team_envs[agent_ID].statusToString(self.world_status),agent_ID,base,ep_num) # update reward
                    else:
                        self.opp_rewards[agent_ID] = self.getReward(
                            self.opp_team_envs[agent_ID].statusToString(self.world_status),agent_ID,base,ep_num) # update reward
                    j+=1
                    self.sync_before_step.wait()


                    # Break if episode done
                    if self.d == True:
                        break

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _start_hfo_server(self):
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
            self.server_port = self.port
            cmd = self.hfo_path + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                  " --log-dir %s --message-size 256"\
                  % (self.fpt, self.untouched, self.num_TA,
                     self.num_OA, self.num_TNPC, self.num_ONPC, self.port,
                     self.offense_on_ball, self.seed, self.config.ball_x_min, self.config.ball_x_max,
                     self.config.ball_y_min, self.config.ball_y_max, self.config.log)
            #Adds the binaries when offense and defense npcs are in play, must be changed to add agent vs binary npc
            if self.num_TNPC > 0:   cmd += " --offense-team %s" \
                % (self.config.left_bin)
            if self.num_ONPC > 0:   cmd += " --defense-team %s" \
                % (self.config.right_bin)
            if not self.config.sync_mode:      cmd += " --no-sync"
            if self.config.fullstate:          cmd += " --fullstate"
            if self.config.determ:      cmd += " --deterministic"
            if self.config.verbose:            cmd += " --verbose"
            if not self.config.rcss_log:  cmd += " --no-logging"
            if self.config.hfo_log:       cmd += " --hfo-logging"
            if self.config.record_lib:             cmd += " --record"
            if self.config.record_serv:      cmd += " --log-gen-pt"
            if self.config.init_env:
                cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                        " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                        " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                        % (self.config.agents_x_min, self.config.agents_x_max, self.config.agents_y_min, self.config.agents_y_max,
                            self.config.change_every_x, self.config.change_agents_x, self.config.change_agents_y,
                            self.config.change_ball_x, self.config.change_ball_y)

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

    def run_envs(self,shared_exps,exp_i,ready,halt,num_updates,history,ep_num):

        # (action_level,feature_level,to_gpu,device,use_viewer,n_training_threads,rcss_log_game,hfo_log_game,num_episodes,replay_memory_size,
        # episode_length,untouched_time,burn_in_iterations,burn_in_episodes, deterministic, num_TA,num_OA,num_TNPC,num_ONPC,offense_team_bin,defense_team_bin,goalie,team_rew_anneal_ep,
        # batch_size,hidden_dim,a_lr,c_lr,tau,explore,final_OU_noise_scale,final_noise_scale,init_noise_scale,num_explore_episodes,D4PG,gamma,Vmax,Vmin,N_ATOMS,
        # DELTA_Z,n_steps,initial_beta,final_beta,num_beta_episodes,TD3,TD3_delay_steps,TD3_noise,I2A,EM_lr,obs_weight,rew_weight,ws_weight,rollout_steps,
        # LSTM_hidden,imagination_policy_branch,SIL,SIL_update_ratio,critic_mod_act,critic_mod_obs,critic_mod_both,control_rand_init,ball_x_min,ball_x_max,
        # ball_y_min,ball_y_max,agents_x_min,agents_x_max,agents_y_min,agents_y_max,change_every_x,change_agents_x,change_agents_y,change_balls_x,change_balls_y,
        # load_random_nets,load_random_every,k_ensembles,current_ensembles,self_play_proba,save_nns,load_nets,initial_models,evaluate,eval_after,eval_episodes,
        # LSTM,LSTM_policy,seq_length,hidden_dim_lstm,lstm_burn_in,overlap,parallel_process,forward_pass,session_path,hist_dir,eval_hist_dir,eval_log_dir,load_path,ensemble_path,t,time_step,discrete_action,
        # log_dir,obs_dim_TA,obs_dim_OA, acs_dim,max_num_experiences,load_same_agent,multi_gpu,data_parallel,play_agent2d,use_preloaded_agent2d,
        # preload_agent2d_path,bl_agent2d,preprocess,zero_critic,cent_critic, record, record_server) = HP

        config = self.config
        if config.record_lib or config.record_serv:
            if os.path.isdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(self.port)):
                file_list = os.listdir(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(self.port))
                [os.remove(os.getcwd() + '/pretrain/pretrain_data/pt_logs_' + str(self.port) + '/' + f) for f in file_list]
            else:
                os.mkdir(os.getcwd() + '/pretrain/pretrain_data//pt_logs_' + str(self.port))

        env = self

        if load_nets:
            maddpg = MADDPG.init_from_save_evaluation(config.initial_models,self.num_TA) # from evaluation method just loads the networks
        else:
            maddpg = MADDPG.init_from_env(env, agent_alg="MADDPG",
                                    adversary_alg= "MADDPG",device=config.device,
                                    gamma=config.gamma,batch_size=config.batch_size,
                                    tau=config.tau,
                                    a_lr=config.a_lr,
                                    c_lr=config.c_lr,
                                    hidden_dim=config.hidden_dim ,discrete_action=config.discrete_action,
                                    vmax=config.vmax,vmin=config.vmin,N_ATOMS=config.n_atoms,
                                    n_steps=config.n_steps,DELTA_Z=config.delta_z,D4PG=config.d4pg,beta=config.init_beta,
                                    TD3=config.td3,TD3_noise=config.td3_noise,TD3_delay_steps=config.td3_delay,
                                    I2A = config.i2a, EM_lr = config.em_lr,
                                    obs_weight = config.obs_w, rew_weight = config.rew_w, ws_weight = config.ws_w, 
                                    rollout_steps = config.roll_steps,LSTM_hidden=config.lstm_hidden,
                                    imagination_policy_branch = config.imag_pol_branch,critic_mod_both=config.cent_q,
                                    critic_mod_act=config.crit_ac, critic_mod_obs= config.crit_obs,
                                    LSTM=config.lstm_crit, LSTM_policy=config.lstm_pol, seq_length=config.seq_length, hidden_dim_lstm=config.hidden_dim_lstm, 
                                    lstm_burn_in=config.burn_in_lstm,overlap=config.overlap,
                                    only_policy=False,multi_gpu=config.multi_gpu,data_parallel=config.data_parallel,preprocess=config.preprocess,
                                    zero_critic=config.zero_crit,cent_critic=config.cent_crit)         
            
        if config.to_gpu:
            maddpg.device = 'cuda'

        if config.multi_gpu:
            if config.num_envs < 5:
                maddpg.torch_device = torch.device("cuda:1")
            else:
                maddpg.torch_device = torch.device("cuda:2")

        current_ensembles = config.current_ensembles
        preload_agent2d_path = ''

        maddpg.prep_training(device=maddpg.device,only_policy=False,torch_device=maddpg.torch_device)

        reward_total = [ ]
        num_steps_per_episode = []
        end_actions = [] 
        team_step_logger_df = pd.DataFrame()
        opp_step_logger_df = pd.DataFrame()

        prox_item_size = self.num_TA*(2*self.team_num_features + 2*self.acs_dim)
        exps = None
        t = 0

        # --------------------------------
        env.launch()
        if config.use_viewer:
            env._start_viewer()       

        time.sleep(3)
        for ep_i in range(0, config.num_ep):
            if to_gpu:
                maddpg.device = 'cuda'

            start = time.time()
            # team n-step
            team_n_step_rewards = []
            team_n_step_obs = []
            team_n_step_acs = []
            team_n_step_next_obs = []
            team_n_step_dones = []
            team_n_step_ws = []


            # opp n-step
            opp_n_step_rewards = []
            opp_n_step_obs = []
            opp_n_step_acs = []
            opp_n_step_next_obs = []
            opp_n_step_dones = []
            opp_n_step_ws = []
            maddpg.prep_policy_rollout(device=maddpg.device,torch_device=maddpg.torch_device)


            
            #define/update the noise used for exploration
            if ep_i < config.burn_in_eps:
                explr_pct_remaining = 1.0
            else:
                explr_pct_remaining = max(0.0, 1.0*config.num_exp_eps - ep_i + config.burn_in_eps) / (config.num_exp_eps)
            beta_pct_remaining = max(0.0, 1.0*config.num_beta_eps - ep_i + config.burn_in_eps) / (config.num_beta_eps)
            
            # evaluation for 10 episodes every 100
            if ep_i % 10 == 0:
                maddpg.scale_noise(config.final_OU_noise_scale + (config.init_noise_scale - config.final_OU_noise_scale) * explr_pct_remaining)
            if ep_i % 100 == 0:
                maddpg.scale_noise(0.0)

            if LSTM:
                maddpg.zero_hidden(1,actual=True,target=True,torch_device=maddpg.torch_device)
            if LSTM_policy:
                maddpg.zero_hidden_policy(1,maddpg.torch_device)
            maddpg.reset_noise()
            maddpg.scale_beta(config.final_beta + (config.init_beta - config.final_beta) * beta_pct_remaining)
            #for the duration of 100 episode with maximum length of 500 time steps
            time_step = 0
            team_kickable_counter = [0] * self.num_TA
            opp_kickable_counter = [0] * self.num_OA
            env.team_possession_counter = [0] * self.num_TA
            env.opp_possession_counter = [0] * self.num_OA
            #reducer = maddpg.team_agents[0].reducer

            # List of tensors sorted by proximity in terms of agents
            sortedByProxTeamList = []
            sortedByProxOppList = []
            for et_i in range(0, config.ep_length):

                if device == 'cuda':
                    # gather all the observations into a torch tensor 
                    torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False).cuda(non_blocking=True,device=maddpg.torch_device)
                                for i in range(self.num_TA)]

                    # gather all the opponent observations into a torch tensor 
                    torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False).cuda(non_blocking=True,device=maddpg.torch_device)
                                for i in range(self.num_OA)]
                    

                else:
                    # gather all the observations into a torch tensor 
                    torch_obs_team = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'team')).T).float(),requires_grad=False)
                                for i in range(self.num_TA)]

                    # gather all the opponent observations into a torch tensor 
                    torch_obs_opp = [Variable(torch.from_numpy(np.vstack(env.Observation(i,'opp')).T).float(),requires_grad=False)
                                for i in range(self.num_OA)] 
        
                # Get e-greedy decision
                if config.explore:
                    team_randoms = misc.e_greedy_bool(env.num_TA,eps = (config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining),device=maddpg.torch_device)
                    opp_randoms = misc.e_greedy_bool(env.num_OA,eps =(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining),device=maddpg.torch_device)
                else:
                    team_randoms = misc.e_greedy_bool(env.num_TA,eps = 0,device=maddpg.torch_device)
                    opp_randoms = misc.e_greedy_bool(env.num_OA,eps = 0,device=maddpg.torch_device)

                # get actions as torch Variables for both team and opp

                team_torch_agent_actions, opp_torch_agent_actions = maddpg.step(torch_obs_team, torch_obs_opp,team_randoms,opp_randoms,parallel=False,explore=config.explore) # leave off or will gumbel sample
                # convert actions to numpy arrays

                team_agent_actions = [ac.cpu().data.numpy() for ac in team_torch_agent_actions]
                #Converting actions to numpy arrays for opp agents
                opp_agent_actions = [ac.cpu().data.numpy() for ac in opp_torch_agent_actions]

                opp_params = np.asarray([ac[0][len(env.action_list):] for ac in opp_agent_actions])

                # this is returning one-hot-encoded action for each team agent
                team_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in team_agent_actions]])
                # this is returning one-hot-encoded action for each opp agent 
                opp_actions = np.asarray([[ac[0][:len(env.action_list)] for ac in opp_agent_actions]])

                

                team_obs =  np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
                opp_obs =  np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T
                
                # use random unif parameters if e_greedy
                team_noisey_actions_for_buffer = team_actions[0]
                team_params = np.array([val[0][len(env.action_list):] for val in team_agent_actions])
                opp_noisey_actions_for_buffer = opp_actions[0]
                opp_params = np.array([val[0][len(env.action_list):] for val in opp_agent_actions])

                team_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in team_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes       
                team_actions_params_for_buffer = np.array([val[0] for val in team_agent_actions])
                opp_agents_actions = [np.argmax(agent_act_one_hot) for agent_act_one_hot in opp_noisey_actions_for_buffer] # convert the one hot encoded actions  to list indexes

                opp_actions_params_for_buffer = np.array([val[0] for val in opp_agent_actions])

                # If kickable is True one of the teammate agents has possession of the ball
                kickable = np.array([env.team_kickable[i] for i in range(env.num_TA)])
                if kickable.any():
                    team_kickable_counter = [tkc + 1 if kickable[i] else tkc for i,tkc in enumerate(team_kickable_counter)]
                    
                # If kickable is True one of the teammate agents has possession of the ball
                kickable = np.array([env.opp_kickable[i] for i in range(env.num_OA)])
                if kickable.any():
                    opp_kickable_counter = [okc + 1 if kickable[i] else okc for i,okc in enumerate(opp_kickable_counter)]
                
                team_possession_counter = [env.get_agent_possession_status(i, env.team_base) for i in range(env.num_TA)]
                opp_possession_counter = [env.get_agent_possession_status(i, env.opp_base) for i in range(env.num_OA)]

                sortedByProxTeamList.append(misc.constructProxmityList(env, team_obs.T, opp_obs.T, team_actions_params_for_buffer, opp_actions_params_for_buffer, env.num_TA, 'left'))
                sortedByProxOppList.append(misc.constructProxmityList(env, opp_obs.T, team_obs.T, opp_actions_params_for_buffer, team_actions_params_for_buffer, env.num_OA, 'right'))

                _,_,_,_,d,world_stat = env.Step(team_agents_actions, opp_agents_actions, team_params, opp_params,team_agent_actions,opp_agent_actions)

                team_rewards = np.hstack([env.Reward(i,'team') for i in range(env.num_TA)])
                opp_rewards = np.hstack([env.Reward(i,'opp') for i in range(env.num_OA)])

                team_next_obs = np.array([env.Observation(i,'team') for i in range(maddpg.nagents_team)]).T
                opp_next_obs = np.array([env.Observation(i,'opp') for i in range(maddpg.nagents_opp)]).T

                
                team_done = env.d
                opp_done = env.d 

                team_n_step_rewards.append(team_rewards)
                team_n_step_obs.append(team_obs)
                team_n_step_next_obs.append(team_next_obs)
                team_n_step_acs.append(team_actions_params_for_buffer)
                team_n_step_dones.append(team_done)
                team_n_step_ws.append(world_stat)

                opp_n_step_rewards.append(opp_rewards)
                opp_n_step_obs.append(opp_obs)
                opp_n_step_next_obs.append(opp_next_obs)
                opp_n_step_acs.append(opp_actions_params_for_buffer)
                opp_n_step_dones.append(opp_done)
                opp_n_step_ws.append(world_stat)
                # ----------------------------------------------------------------
                # Reduce size of obs

                time_step += 1
                t += 1

                if t%3000 == 0:
                    team_step_logger_df.to_csv(hist_dir + '/team_%s.csv' % history)
                    opp_step_logger_df.to_csv(hist_dir + '/opp_%s.csv' % history)
                            
                team_episode = []
                opp_episode = []

                if d == 1 and et_i >= (seq_length-1): # Episode done 
                    n_step_gammas = np.array([[config.gamma**step for a in range(env.num_TA)] for step in range(config.n_steps)])
                #NOTE: Assume M vs M and critic_mod_both == True
                    if critic_mod_both:
                        team_all_MC_targets = []
                        opp_all_MC_targets = []
                        MC_targets_team = np.zeros(env.num_TA)
                        MC_targets_opp = np.zeros(env.num_OA)
                        for n in range(et_i+1):
                            MC_targets_team = team_n_step_rewards[et_i-n] + MC_targets_team*config.gamma
                            team_all_MC_targets.append(MC_targets_team)
                            MC_targets_opp = opp_n_step_rewards[et_i-n] + MC_targets_opp*config.gamma
                            opp_all_MC_targets.append(MC_targets_opp)
                        for n in range(et_i+1):
                            n_step_targets_team = np.zeros(env.num_TA)
                            n_step_targets_opp = np.zeros(env.num_OA)
                            if (et_i + 1) - n >= config.n_steps: # sum n-step target (when more than n-steps remaining)
                                n_step_targets_team = np.sum(np.multiply(np.asarray(team_n_step_rewards[n:n+config.n_steps]),(n_step_gammas)),axis=0)
                                n_step_targets_opp = np.sum(np.multiply(np.asarray(opp_n_step_rewards[n:n+config.n_steps]),(n_step_gammas)),axis=0)

                                n_step_next_ob_team = team_n_step_next_obs[n - 1 + config.n_steps]
                                n_step_done_team = team_n_step_dones[n - 1 + config.n_steps]

                                n_step_next_ob_opp = opp_n_step_next_obs[n - 1 + config.n_steps]
                                n_step_done_opp = opp_n_step_dones[n - 1 + config.n_steps]
                            else: # n-step = MC if less than n steps remaining
                                n_step_targets_team = team_all_MC_targets[et_i-n]
                                n_step_next_ob_team = team_n_step_next_obs[-1]
                                n_step_done_team = team_n_step_dones[-1]

                                n_step_targets_opp = opp_all_MC_targets[et_i-n]
                                n_step_next_ob_opp = opp_n_step_next_obs[-1]
                                n_step_done_opp = opp_n_step_dones[-1]
                            if D4PG:
                                default_prio = 5.0
                            else:
                                default_prio = 3.0
                            priorities = np.array([np.zeros(config.k_ensembles) for i in range(env.num_TA)])
                            priorities[:,current_ensembles] = 5.0
                            if SIL:
                                SIL_priorities = np.ones(env.num_TA)*default_prio
                            

                            exp_team = np.column_stack((np.transpose(team_n_step_obs[n]),
                                                team_n_step_acs[n],
                                                np.expand_dims(team_n_step_rewards[n], 1),
                                                np.expand_dims([n_step_done_team for i in range(env.num_TA)], 1),
                                                np.expand_dims(team_all_MC_targets[et_i-n], 1),
                                                np.expand_dims(n_step_targets_team, 1),
                                                np.expand_dims([team_n_step_ws[n] for i in range(env.num_TA)], 1),
                                                priorities,
                                                np.expand_dims([default_prio for i in range(env.num_TA)],1)))


                            exp_opp = np.column_stack((np.transpose(opp_n_step_obs[n]),
                                                opp_n_step_acs[n],
                                                np.expand_dims(opp_n_step_rewards[n], 1),
                                                np.expand_dims([n_step_done_opp for i in range(env.num_OA)], 1),
                                                np.expand_dims(opp_all_MC_targets[et_i-n], 1),
                                                np.expand_dims(n_step_targets_opp, 1),
                                                np.expand_dims([opp_n_step_ws[n] for i in range(env.num_OA)], 1),
                                                priorities,
                                                np.expand_dims([default_prio for i in range(env.num_TA)],1)))
                    
                            exp_comb = np.expand_dims(np.vstack((exp_team, exp_opp)), 0)

                            if exps is None:
                                exps = torch.from_numpy(exp_comb)
                            else:
                                exps = torch.cat((exps, torch.from_numpy(exp_comb)),dim=0)
                        
                        prox_team_tensor = misc.convertProxListToTensor(sortedByProxTeamList, env.num_TA, prox_item_size)
                        prox_opp_tensor = misc.convertProxListToTensor(sortedByProxOppList, env.num_OA, prox_item_size)
                        comb_prox_tensor = torch.cat((prox_team_tensor, prox_opp_tensor), dim=1)
                        # Fill in values for zeros for the hidden state
                        exps = torch.cat((exps[:, :, :], torch.zeros((len(exps), env.num_TA*2, config.hidden_dim_lstm*4), dtype=exps.dtype), comb_prox_tensor.double()), dim=2)
                        #maddpg.get_recurrent_states(exps, obs_dim_TA, acs_dim, env.num_TA*2, hidden_dim_lstm,maddpg.torch_device)
                        shared_exps[int(ep_num[config.num_envs].item())][:len(exps)] = exps
                        exp_i[int(ep_num[config.num_envs].item())] += et_i
                        ep_num[config.num_envs] += 1
                        del exps
                        exps = None
                        torch.cuda.empty_cache()

                    #############################################################################################################################################################
                    # push exp to queue
                    # log
                    if ep_i > 1:
                        team_avg_rew = [np.asarray(team_n_step_rewards)[:,i].sum()/float(et_i) for i in range(env.num_TA)] # divide by time step?
                        team_cum_rew = [np.asarray(team_n_step_rewards)[:,i].sum() for i in range(env.num_TA)]
                        opp_avg_rew = [np.asarray(opp_n_step_rewards)[:,i].sum()/float(et_i) for i in range(env.num_TA)]
                        opp_cum_rew = [np.asarray(opp_n_step_rewards)[:,i].sum() for i in range(env.num_TA)]


                        team_step_logger_df = team_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(tkc/time_step)*100 for tkc in team_kickable_counter],
                                                            'possession_percentages': [(tpc/time_step)*100 for tpc in team_possession_counter],
                                                            'average_reward': team_avg_rew,
                                                            'cumulative_reward': team_cum_rew},
                                                            ignore_index=True)

                        
                        opp_step_logger_df = opp_step_logger_df.append({'time_steps': time_step, 
                                                            'why': env.opp_team_envs[0].statusToString(world_stat),
                                                            'agents_kickable_percentages': [(okc/time_step)*100 for okc in opp_kickable_counter],
                                                            'possession_percentages': [(opc/time_step)*100 for opc in opp_possession_counter],
                                                            'average_reward': opp_avg_rew,
                                                            'cumulative_reward': opp_cum_rew},
                                                            ignore_index=True)


                    # Launch evaluation session
                    if ep_i > 1 and ep_i % config.eval_after == 0 and config.eval:
                        thread.start_new_thread(launch_eval,(
                            [load_path + ("agent_%i/model_episode_%i.pth" % (i,ep_i)) for i in range(env.num_TA)], # models directory -> agent -> most current episode
                            eval_episodes,eval_log_dir,eval_hist_dir + "/evaluation_ep" + str(ep_i),
                            7000,env.num_TA,env.num_OA,config.ep_length,config.device,config.use_viewer,))
                    if halt.all(): # load when other process is loading buffer to make sure its not saving at the same time
                        ready[config.num_envs] = 1
                        current_ensembles = maddpg.load_random_ensemble(side='team',nagents=env.num_TA,models_path = config.ensemble_path,load_same_agent=config.load_same_agent) # use for per ensemble update counter

                        if config.agent2d and config.preloaded_agent2d:
                            maddpg.load_agent2d(side='opp',load_same_agent=config.load_same_agent,models_path=preload_agent2d_path,nagents=env.num_OA)
                        elif play_agent2d:
                            maddpg.load_agent2d(side='opp',models_path =config.session_path +"models/",load_same_agent=config.load_same_agent,nagents=env.num_OA)  
                        else:
                                
                            if np.random.uniform(0,1) > self_play_proba: # self_play_proba % chance loading self else load an old ensemble for opponent
                                maddpg.load_random(side='opp',nagents =env.num_OA,models_path =config.load_path,load_same_agent=config.load_same_agent)
                                pass
                            else:
                                maddpg.load_random_ensemble(side='opp',nagents = num_OA,models_path = config.ensemble_path,load_same_agent=config.load_same_agent)
                                pass

                        if bl_agent2d:
                            maddpg.load_agent2d(side='team',load_same_agent=config.load_same_agent,models_path=preload_agent2d_path,nagents=env.num_OA)

                        while halt.all():
                            time.sleep(0.1)
                        total_dim = (obs_dim_TA + self.acs_dim + 5) + config.k_ensembles + 1 + (config.hidden_dim_lstm*4) + prox_item_size
                        ep_num.copy_(torch.zeros_like(ep_num,requires_grad=False))
                        [s.copy_(torch.zeros(config.max_num_exps,2*env.num_TA,total_dim)) for s in shared_exps[:int(ep_num[config.num_envs].item())]] # done loading
                        del exps
                        exps = None

                    end = time.time()
                    print(end-start)

                    break
                elif d:
                    break            
                    
                team_obs = team_next_obs
                opp_obs = opp_next_obs

import numpy as np
import time
import threading
import pandas as pd
import math
import os, subprocess, time, signal
import HFO
import HFO.hfo.hfo as hfo
import misc
# from .HFO.hfo import hfo
# from .HFO import get_config_path, get_hfo_path, get_viewer_path

# from utils import misc as misc
from torch.autograd import Variable
import torch

possession_side = 'N'

class rc_env:
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

    def __init__(self, config, port):
        self.config = config
        self.untouched = config['untouched']
        self.goalie = config['goalie']
        self.port = port
        self.hfo_path = HFO.get_hfo_path()
        self.seed = np.random.randint(1000)
        self.viewer = None

        self.num_left = config['num_left']
        self.num_right = config['num_right']
        self.num_leftBot = config['num_l_bot']
        self.num_rightBot = config['num_r_bot']

        # params for low level actions
        num_action_params = 5 # 2 for dash and kick 1 for turn and tackle
        self.team_action_params = np.asarray([[0.0]*num_action_params for i in range(config.num_left)])
        self.opp_action_params = np.asarray([[0.0]*num_action_params for i in range(config.num_right)])
        if config['action_level'] == 'low':
            #                   pow,deg   deg       deg         pow,deg    
            #self.action_list = [hfo.DASH, hfo.TURN, hfo.TACKLE, hfo.KICK]
            self.action_list = [hfo.DASH, hfo.TURN, hfo.KICK]
            self.kick_actions = [hfo.KICK] # actions that require the ball to be kickable
        elif config['action_level'] == 'high':
            self.action_list = [hfo.DRIBBLE, hfo.SHOOT, hfo.REORIENT, hfo.GO_TO_BALL, hfo.MOVE]
            self.kick_actions = [hfo.DRIBBLE, hfo.SHOOT, hfo.PASS] # actions that require the ball to be kickable

        self.fpt = config['ep_length']

        self.act_lvl = config['action_level']
        self.feat_lvl = config['feature_level']

        if config['feature_level'] == 'low':
            #For new obs reorganization without vailds, changed hfo obs from 59 to 56
            self.left_features = 56 + 13*(self.num_left-1) + 12*self.num_right + 4 + 1 + 2 + 1  + 8
            self.right_features = 56 + 13*(self.num_right-1) + 12*self.num_left + 4 + 1 + 2 + 1 + 8
        elif config.fl == 'high':
            self.left_features = (6*self.num_left) + (3*self.num_right) + (3*self.num_rightBot) + 6
            self.right_features = (6*self.num_right) + (3*self.num_left) + (3*self.num_rightBot) + 6
        elif config.fl == 'simple':
            # 16 - land_feats + 12 - basic feats + 6 per (team/opp)
            self.left_features = 28 + (6 * ((self.num_left-1) + self.num_right)) + 8
            self.right_features = 28 + (6 * ((self.num_right-1) + self.num_left)) + 8

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

        # Left side actions, obs, rewards
        self.left_actions = np.array([2]*self.num_left)
        self.left_actions_OH = np.empty([self.num_left, 8],dtype=float)
        self.left_obs = np.empty([self.num_left,self.left_features],dtype=float)
        self.left_obs_previous = np.empty([self.num_left,self.left_features],dtype=float)
        self.left_rewards = np.zeros(self.num_left)

        # Right side actions, obs, rewards
        self.right_actions = np.array([2]*self.num_right)
        self.right_obs = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_obs_previous = np.empty([self.num_right,self.right_features],dtype=float)
        self.right_actions_OH = np.empty([self.num_right, 8],dtype=float)
        self.right_rewards = np.zeros(self.num_right)

        self.world_status = 0
        
        self.left_base = 'base_left'
        self.right_base = 'base_right'
        
    def launch(self):
        self._start_hfo_server()
        self.left_envs = [hfo.HFOEnvironment() for i in range(self.num_left)]
        self.right_envs = [hfo.HFOEnvironment() for i in range(self.num_right)]
        
        # Create thread(s) for left side
        for i in range(self.num_left):
            if i == 0:
                print("Connecting player %i" % i , "on team %s to the server" % self.left_base)
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.left_base,
                                                self.goalie,i,self.fpt,self.act_lvl,))
                t.start()
            else:
                print("Connecting player %i" % i , "on team %s to the server" % self.left_base)
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.left_base,
                                                False,i,self.fpt,self.act_lvl,))
                t.start()

            time.sleep(1.5)
        
        for i in range(self.num_right):
            if i == 0:
                print("Connecting player %i" % i , "on Opponent %s to the server" % self.right_base)
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.right_base,
                                                self.goalie,i,self.fpt,self.act_lvl,))
                t.start()
            else:
                print("Connecting player %i" % i , "on Opponent %s to the server" % self.right_base)
                t = threading.Thread(target=self.connect, args=(self.port,self.feat_lvl, self.right_base,
                                                False,i,self.fpt,self.act_lvl,))
                t.start()

            time.sleep(1.5)
        print("All players connected to server")
        self.start = True

    def Observation(self,agent_id,side):
        if side == 'left':
            return self.left_obs[agent_id]
        elif side == 'right':
            return self.right_obs[agent_id]

    def Reward(self,agent_id,side):
        if side == 'left':
            return self.left_rewards[agent_id]
        elif side == 'right':
            return self.right_rewards[agent_id]


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
    def getReward(self,s,agentID,base,ep_num):
        reward=0.0
        return reward

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

        config_dir = HFO.get_config_path() 
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
        cmd = HFO.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)


import random 
import numpy as np
import hfo
import time
import _thread as thread


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
        
        
    Todo:
        * Functionality for synchronizing team actions with opponent team actions
        """
    
    


    
    
    def __init__(self, num_TA,num_OA,num_ONPC,base,goalie,num_trials,fpt,feat_lvl,act_lvl):
                
        self.num_TA = num_TA 
        self.num_OA = num_OA
        self.num_ONPC = num_ONPC
        
        if feat_lvl == 'low':
            # TODO find **ACCURATE** num_features for low level
            self.num_features = 0
        elif feat_lvl == 'high':
            self.num_features = (6*num_TA) + (3*num_OA) + (3*num_ONPC) + 6
        

        # Create env for each teammate
        self.team_envs = [hfo.HFOEnvironment() for i in range(num_TA)]
        self.opp_team_envs = [hfo.HFOEnvironment() for i in range(num_OA)]


        self.d = False
        
        self.start = False
        self.loading_flag = True
        self.loading = np.array([1]*num_TA)
        self.wait = np.array([0]*num_TA)
        self.wait_flag = False
        
        
        # Initialization of mutable lists to be passsed to threads
        self.team_actions = np.array([3]*num_TA)
        self.action_list = [hfo.DRIBBLE, hfo.SHOOT, hfo.REORIENT, hfo.GO_TO_BALL ]
        self.team_should_act = np.array([0]*num_TA)
        self.team_should_act_flag = False
        
        self.team_rewards = np.zeros(num_TA)
        self.team_obs = np.empty([num_TA,self.num_features],dtype=object)

        self.opp_actions = np.zeros(num_OA)
        self.opp_should_act = np.array([0]*num_OA)
        self.opp_should_act_flag = False

        self.opp_obs = np.empty([num_OA,5],dtype=object)
        self.opp_rewards = np.zeros(num_OA)

        self.world_status = 0
        


        

        # Create thread for each teammate
        for i in range(num_TA):
            print("Connecting player %i" % i , "on team %s to the server" % base)
            thread.start_new_thread(self.connect,(feat_lvl, base, 
                                             False,i,))
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


    def Step(self,actions,side):
        """ Performs each agents' action from actions and returns world_status
        
        Args:
            actions (list of ints); List of integers corresponding to the action each agent will take
            side (str): Which team agent belongs to. ('team', 'opp')
            
        Returns:
            Status of HFO World
            
        Todo:
            * Add functionality for opp team
            
        """
        [self.Queue_action(i,actions[i],side) for i in range(len(actions))]
        
        return np.asarray(self.team_obs),self.team_rewards,self.d
        
        
        
    def Queue_action(self,agent_id,action,side):
        """ Queues an action on agent, and if all agents have received action instructions performs the actions.
        
        Args:
            agent_id (int): Agent to receive observation from. (0-11)
            side (str): Which team agent belongs to. ('team', 'opp')
        

        Todo:
            * Add halting until agents from both teams have received action 
            
        """
        # Function is running too fast for agents to act so wait until the agents have acted

        while not self.wait_flag:
            time.sleep(0.0001)
        self.wait[agent_id] = False
        
        if side == 'team':
            self.team_actions[agent_id] = action
        elif side == 'opp':
            self.opp_actions[agent_id] = action

        # Sets action flag for this agent to True.
        # Sets team action flag to True allowing threads to take action only 
        #once each thread's personal action flag is set to true.
        self.team_should_act[agent_id] = True
        if self.team_should_act.all():
            self.team_should_act_flag = True
            while self.team_should_act_flag:
                time.sleep(0.0001)
        
        if not self.wait.any():
            self.wait_flag = False
        
        time.sleep(0.001) ### *** without this sleep function the process crashes. specifically, 0.001 



    # Engineered Reward Function
    def getReward(self,s):    
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
    def connect(self,feat_lvl, base, goalie, agent_ID):
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
        max_epLength = 1000
        num_episodes = 20000


        # Once all agents have been loaded,
        # wait for action command, take action, update: obs, reward, and world status
        while(True):
            while(self.start):



                j = 0 # j to maximum episode length
                d = False
                self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # Get initial state
                print('length of observation: ', len(self.team_obs[agent_ID]))
                while j < max_epLength:
                    j+=1
                    # If the action flag is set, each thread takes its action
                    # Sets its personal flag to false, such that if all agents have taken their action 
                    # and have updated values we may return to the Step function call
                    # (This synchronizes all agents actions so that they halt until other agents have taken theirs)

                    self.wait[agent_ID] = True
                    while not self.wait_flag:
                        if(self.wait.all()):
                            self.wait_flag = True
                        else:
                            time.sleep(0.0001)

                    if self.loading[agent_ID]:
                        print("Agent %i loaded" % agent_ID)
                        self.loading[agent_ID] = False
                    if (not self.loading.any()) and self.loading_flag:
                        self.loading_flag = False
                        print("Loaded and ready for action order")




                    while not self.team_should_act_flag and not self.team_should_act[agent_ID]:
                        time.sleep(0.0001)

                    self.team_envs[agent_ID].act(self.action_list[self.team_actions[agent_ID]]) # take the action
                    self.world_status = self.team_envs[agent_ID].step() # update world
                    self.team_rewards[agent_ID] = self.getReward(
                       self.team_envs[agent_ID].statusToString(self.world_status)) # update reward
                    self.team_obs[agent_ID] = self.team_envs[agent_ID].getState() # update obs
                    self.team_should_act[agent_ID] = False # Set personal action flag as done

                    # Halts until all threads have taken their action and resets flag to off
                    while(self.team_should_act.any()):
                        time.sleep(0.0001)
                    self.team_should_act_flag = False


                    # Break if episode done
                    if self.world_status == hfo.IN_GAME:
                        self.d = False
                    else:
                        self.d = True

                    if self.d == True:
                        break

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    

   
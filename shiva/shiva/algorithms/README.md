# Algorithm Folder
## Config Requirements
## Contents
*   Algorithm.py
*   ContinousDDPGAlgorithm.py
*   DQNAlgorithm.py
*   DaggerAlgorithm.py
*   SupervisedAlgorithm.py
*   init.py  

##  Algorithm.py

##  ContinousDDPGAlgorithm.py
##  DQNAlgorithm.py
##  DaggerAlgorithm.py
*   Supports Discrete and Continuous Action Space. 
*   init(self,obs_space, acs_space, action_space_discrete,action_space_continuous,configs):
    -   ***DESCRIPTION***
        +   Initlizes the Dagger Algorithm. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance. |
        |   **obs_space**   |Feed in the observation space. |
        |    **acs_space**   | Feed in the action space.  |
        |   **action_space_discrete**   | Feed in the Discrete Action Space.  |
        |     **action_space_continous**   | Feed in the Continous Action Space. |
  
*   update(self, imitation_agent, expert_agent, minibatch, step_n)
    -   ***DESCRIPTION***
        +   Calculates Trajectories from imation policy and inital policy, but allows for the new agent to explore new observations; in addition, to what the expert agent has led us to. Then we calculate the loss between the imitation agent and expert agent and using a desired loss calcualtion from the config. then we optimize. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance. |
        |   **imitation_agent**   |Agent that we are making, updating and learning.  |
        |    **expert_agent**   | The agent that we are imitating from.  |
        |   **minibatch**   | The Minibatch number.  |
        |     **step_n**   | The current step number. |
   
*   get_action(self, agent, observation,step_n)
    -   ***DESCRIPTION***
        +   Gets the action taken from the imitation agent from the observation space that has been fed. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **agent**   | Get an action from the imitation agent.  |
        |   **observation**  | Get an observation from the action passed to the agent.  |
        |     **step_n**   | The current step number. |
*   find_best_action(self,network, observation: np.ndarray)
    -   ***DESCRIPTION***
        +   Gets the best action from an array of observations that has been fed. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **network**  | Pass the current network of the imitation agent.  |
        |   **observation: np.ndarray**  | Pass the observation space as an numpy array.|
        
  
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   ***DESCRIPTION***
        +   Gets the best expert action from the expert from the array of observations.
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **network**  | The network of the expert agent  |
        |   **observation: np.ndarray**  | Pass the observation space as an numpy array.|

*   get_loss(self)
    -   ***DESCRIPTION***
        +   Get the loss value to partciular agent, and calculation based on what is set at the config.
    -   Get the loss value that has been assigned to the agent. 



##  SupervisedAlgorithm.py
*   Supports Discrete and Continuous Action Space. 
    -   ***DESCRIPTION***
        +   Initializes the Dagger Algorithm.
*   init(self,obs_space, acs_space, action_space_discrete,action_space_continuous,configs):
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance. |
        |   **obs_space**   |Feed in the observation space. |
        |    **acs_space**   | Feed in the action space.  |
        |   **action_space_discrete**   | Feed in the Discrete Action Space.  |
        |     **action_space_continous**   | Feed in the Continous Action Space. |
*   update(self, imitation_agent, expert_agent, minibatch, step_n)
    -   ***DESCRIPTION***
        +   Collecting Trajectories from the expert agent on a replay buffer, and calculate loss between expert agent actions, based on the selected settings in the config file. Then we optimize. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance. |
        |   **imitation_agent**   |Agent that we are making, updating and learning.  |
        |    **expert_agent**   | The agent that we are imitating from.  |
        |   **minibatch**   | The Minibatch number.  |
        |     **step_n**   | The current step number. |
   
*   get_action(self, agent, observation,step_n)
    -   ***DESCRIPTION***
        +   Gets the action taken from the imitation agent from the observation space that has been fed. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **agent**   | Get an action from the imitation agent.  |
        |   **observation**  | Get an observation from the action passed to the agent.  |
        |     **step_n**   | The current step number. |
*   find_best_action(self,network, observation: np.ndarray)
    -   ***DESCRIPTION***
        +   Gets the best action from an array of observations that has been fed. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **network**  | Pass the current network of the imitation agent.  |
        |   **observation: np.ndarray**  | Pass the observation space as an numpy array.|
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   ***DESCRIPTION***
        +   Gets the best expert action from the expert from the array of observations.
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |    **self**   | Refers to the Current Instance |
        |    **network**  | The network of the expert agent  |
        |   **observation: np.ndarray**  | Pass the observation space as an numpy array.|
*   get_loss(self)
    -   ***DESCRIPTION***
        +   Get the loss value to partciular agent, and calculation based on what is set at the config. 
    -   Get the loss value that has been assigned to the agent. 
    
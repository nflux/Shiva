# Algorithms
## Config Requirements
Are specified in detail below.
## Contents
*   Algorithm (Abstract)
    *   [Link to Code](../shiva/algorithms/Algorithm.py)
*   ContinousDDPGAlgorithm
    *   [Link to Code](../shiva/algorithms/ContinuousDDPGAlgorithm.py)
*   DQNAlgorithm
    *   [Link to Code](../shiva/algorithms/DQNAlgorithm.py)
*   DaggerAlgorithm
    *   [Link to Code](../shiva/algorithms/DaggerAlgorithm.py)
*   DiscreteDDPGAlgorithm
    *   [Link to Code](../shiva/algorithms/DiscreteDDPGAlgorithm.py)
*   ParameterizedDDPGAlgorithm
    *   [Link to Code](../shiva/algorithms/ParametrizedDDPGAlgorithm.py)
*   PPOAlgorithm
    *   [Link to Code](../shiva/algorithms/PPOAlgorithm.py)
*   SupervisedAlgorithm
    *   [Link to Code](../shiva/algorithms/SupervisedAlgorithm.py)
*   init
    *   [Link to Code](../shiva/algorithms/__init__.py)

##  Algorithm (Abstract)
[Link to Code](../shiva/algorithms/Algorithm.py)

All the algorithms inherit from this abstract class. Learn more about Shiva's abstract classes [here](../docs/Abstract-Classes.md).
##  ContinuousDDPGAlgorithm
[Link to Code](../shiva/algorithms/ContinuousDDPGAlgorithm.py)
### Config Set Up     
```
[Algorithm]
algorithm='DDPG'
type="ContinuousDDPGAlgorithm"
exploration_steps=2000
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.9999
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
tau=0.99
```
##  DQNAlgorithm
[Link to Code](../shiva/algorithms/DQNAlgorithm.py)
### Config Set Up     
```
[Algorithm]
type='DQNAlgorithm'
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=False
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
```
##  DaggerAlgorithm
[Link to Code](../shiva/algorithms/DaggerAlgorithm.py)
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

### Config Set Up     
```
[Algorithm]
type1='SupervisedAlgorithm'
type2='DaggerAlgorithm'
replay_buffer=True
learning_rate=0.01
optimizer='Adam'
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
```

##  DiscreteDDPGAlgorithm
[Link to Code](../shiva/algorithms/DiscreteDDPGAlgorithm.py)
### Config Set Up     
```
[Algorithm]
algorithm='DDPG'
type="DiscreteDDPGAlgorithm"
exploration_steps=10_000
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
tau=0.999
```
## ParameterizedDDPGAlgorithm
[Link to Code](../shiva/algorithms/ParametrizedDDPGAlgorithm.py)
### Config Set Up     
```
[Algorithm]
algorithm='DDPG'
type="ParametrizedDDPGAlgorithm"
exploration_steps=10_000
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
tau=0.0001
```
## PPO
[Link to Code](../shiva/algorithms/PPOAlgorithm.py)
### Config Set Up     
```
[Algorithm]
algorithm='PPO'
type="PPOAlgorithm"
episodes_train = 10
old_policy_update_interval = 1000
update_epochs = 5
exploration_steps=0
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.9
beta=0.1
epsilon_clip = 0.1
epsilon_start= 1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
tau=0.99
```
##  SupervisedAlgorithm.py
[Link to Code](../shiva/algorithms/SupervisedAlgorithm.py)
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
    ### Config Set Up     
```
[Algorithm]
type1='SupervisedAlgorithm'
type2='DaggerAlgorithm'
replay_buffer=True
learning_rate=0.01
optimizer='Adam'
loss_function='MSELoss'
regularizer=0
recurrence=0
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
```

## init
[Link to Code](../shiva/algorithms/__init__.py)

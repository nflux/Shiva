# Algorithm Folder
## Config Requirements
## Contents
*   Algorithm.py
*   ContinousDDPGAlgorithm.py
*   DaggerAlgorithm.py
*   DQNAlgorithm.py
*   SupervisedAlgorithm.py
*   init.py  

##  Algorithm.py

##  ContinousDDPGAlgorithm.py

##  DaggerAlgorithm.py
*   update(self, imitation_agent, expert_agent, minibatch, step_n)
    -   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **imitation_agent**
        +   Agent that we are making, updating and learning. 
    -   **expert_agent**
        +   The agent that we are imitating from.
    -   **minibatch**
        +   The Minibatch number.
    -   **step_n**
        +   The current step number.
*   get_action(self, agent, observation,step_n)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   Get an action from the imitation agent.
    -   **observation**
        +   Get an observation from the action passed to the agent.
    -   **step_n**
        +   The current Step Number.
*   find_best_action(self,network, observation: np.ndarray)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **network**
        +   Pass the current network of the imitation agent. 
    -   **observation: np.ndarray**
        +   Pass the observation space as an numpy array.
*   find_best_expert_action(self, network, observation: np.ndarray)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **network**
        +   The network of the expert agent.
    -   **observaton: np.ndarray**
         +   Pass the observation space as an numpy array.
*   get_lost(self)
-   ***DESCRIPTION***
        +
    -   Get the loss value that has been assigned to the agent. 

##  DQNAlgorithmpy

##  SupervisedAlgorithm.py
*   update(self, imitation_agent, expert_agent, minibatch, step_n)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **imitation_agent**
        +   Agent that we are making, updating and learning. 
    -   **expert_agent**
        +   The agent that we are imitating from.
    -   **minibatch**
        +   The Minibatch number.
    -   **step_n**
        +   The current Step Number.
*   get_action(self, agent, observation,step_n)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   Get an action from the imitation agent.
    -   **observation**
        +   Pass the observation space as an numpy array.
    -   **step_n**
        +   The current Step Number.
*   find_best_action(self,network, observation: np.ndarray)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **network**
    -   **observation: np.ndarray**
*   find_best_expert_action(self, network, observation: np.ndarray)
-   ***DESCRIPTION***
        +
    -   **self**
        +   Refers to the Current Instance
    -   **network**
    -   **observaton: np.ndarray**
*   get_lost(self)
-   ***DESCRIPTION***
        +
    -   Get the loss value that has been assigned to the agent. 
    
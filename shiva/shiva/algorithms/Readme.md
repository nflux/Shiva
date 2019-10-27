# Algorithm Folder
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
    -   **self**
        +   Refers to the Current Instance
    -   **imitation_agent**
        +   Agent that we are making, updating and learning. 
    -   **expert_agent**
        +   The agent that we are imitating from.
    -   **minibatch**
        +   The Minibatch nubmer.
    -   **step_n**
        +   The current step number.
*   get_action(self, agent, observation,step_n)
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   Get an action from the imitation agent.
    -   **observation**
        +   Get an observation from the action passed to the agent.
    -   **step_n**
        +   The current Step Number.
*   find_best_action(self,network, observation: np.ndarray)
    -   **self**
        +   Refers to the Current Instance
    -   **network**
        +   Pass the current network of the imitation agent. 
    -   **observation: np.ndarray**
        +   Pass the observation space as an numpy array.
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   **self**
        +   Refers to the Current Instance
    -   **network**
        +   The network of the expert agent.
    -   **observaton: np.ndarray**
         +   Pass the observation space as an numpy array.
*   get_lost(self)
##  DQNAlgorithmpy

##  SupervisedAlgorithm.py
*   update(self, imitation_agent, expert_agent, minibatch, step_n)
    -   **self**
        +   Refers to the Current Instance
    -   **imitation_agent**
    -   **expert_agent**
    -   **minibatch**
    -   **step_n**
*   get_action(self, agent, observation,step_n)
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
    -   **observation**
    -   **step_n**
*   find_best_action(self,network, observation: np.ndarray)
    -   **self**
        +   Refers to the Current Instance
    -   **network**
    -   **observation: np.ndarray**
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   **self**
        +   Refers to the Current Instance
    -   **network**
    -   **observaton: np.ndarray**
*   get_lost(self)
    
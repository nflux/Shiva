# Algorithm Folder
## Contents
*   Algorithm.py
*   ContinousDDPGAlgorithm.py
*   DaggerAlgorithm.py
*   DQNAlgorithm.py
*   SupervisedAlgorithm.py
*   init.py  

##  Algorithm.py

##  ContonousDDPGAlgorithm.py

##  DaggerAlgorithm.py
*   update(self, imitation_agent, expert_agent,minibatch, step_n)
    -   self
        +   Refers to the Current Instance
    -   imitation_agent
    -   expert_agent
    -   minibatch
    -   step_n
*   get_action(self, agent, observation,step_n)
    -   self
        +   Refers to the Current Instance
    -   agent
    -   observation
    -   step_n
*   find_best_action(self,network, observation: np.ndarray)
    -   self
        +   Refers to the Current Instance
    -   network
    -   observation: np.ndarray
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   self
        +   Refers to the Current Instance
    -   network
    -   observaton: np.ndarray
*   get_lost(self)
##  DQNAlgorithmpy

##  SupervisedAlgorithm.py
*   update(self, imitation_agent, expert_agent,minibatch, step_n)
    -   self
        +   Refers to the Current Instance
    -   imitation_agent
    -   expert_agent
    -   minibatch
    -   step_n
*   get_action(self, agent, observation,step_n)
    -   self
        +   Refers to the Current Instance
    -   agent
    -   observation
    -   step_n
*   find_best_action(self,network, observation: np.ndarray)
    -   self
        +   Refers to the Current Instance
    -   network
    -   observation: np.ndarray
*   find_best_expert_action(self, network, observation: np.ndarray)
    -   self
        +   Refers to the Current Instance
    -   network
    -   observaton: np.ndarray
*   get_lost(self)
    
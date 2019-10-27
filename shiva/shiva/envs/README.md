# Environments Folder
## Contents
*   Environment.py
*   GymContinuousEnvironment.py
*   GymDiscreteEnvironment.py
*   RobocupDDPGEnvironment.py
*   init.py  

##  Environment.py 
*   **DESCRIPTION**: This is the abstract class that the other environments will inherit from.
*   step(self,actions)
    -   **self**
        +   Refers to the Current Instance
    -   **actions**
        +   
*   get_observation(self, agent)
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   
*   get_observations(self)
    -   **self**
        +   Refers to the Current Instance
*   get_action(self, agent)
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   
*   get_actions(self)
    -   **self**
        +   Refers to the Current Instance 
*   get_reward(self, agent)
    -   **self**
        +   Refers to the Current Instance
    -   **agent**
        +   
*   get_rewards(self)
    -   **self**
        +   Refers to the Current Instance
*   get_observation_space(self)
    -   **self**
        +   Refers to the Current Instance
*   get_action_space(self)
    -   **self**
        +   Refers to the Current Instance 
*   get_current_step(self)
    -   **self**
        +   Refers to the Current Instance 
*   reset(self)
    -   **self**
        +   Refers to the Current Instance 
*   load_viewer(self)
    -   **self**
        +   Refers to the Current Instance  
*   normalize_reward(self)
    -   **self**
        +   Refers to the Current Instance  

##  GymContinuousEnvironment.py


##  GymDiscreteEnvironment.py

##  RobocupDDPGEnvironment.py

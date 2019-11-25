#   Agents Folder
## Config Requirements
## Contents
*   Agent.py
*   DDPGAgent.py
*   DQNAgent.py
*   ImitationAgent.py
*   init_agent.py

##  Agent.py
##  DDPGAgent.py
##  DQNAgent.py
##  ImitationAgent.py
##  init_agent.py
##  ParameterizedDDPGAgent.py 
##  PPOAgent.py


##  ImitationAgent.py
*   Creates an imitation agent
*   init(self, id, obs_dim, acs_discrete, acs_continuous, agent_config,net_config)
    -   ***Description***
        +   Initalizes the Imitation Agent. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |   **self**    |     The Instance of ImitationAgent.   |
        |   **id**      |  Assign the ID of Agent.     |
        |   **obs_dim** |   Pass in the Observation Dimensions    |
        |   **acs_discrete**    |   Pass the Discrete Action Space.|
        |   **acs_continous**   |Pass the Continous Action Space|
        |   **agent_config**    | Passes in the Agent Section of the Config File.     |

*   find_best_action(self,network,observation)
    -   ***Description***
        +   Get the best action from the agent. 
    -   |   Variables   |   Description   |
        |       ---     |       ---       |
        |   **self**    |     The Instance of ImitationAgent.   |
        |   **network**    |     The network/policy of ImitationAgent.   |
        |   **observation**    |     Pass the observation to get an action.   |
        
 
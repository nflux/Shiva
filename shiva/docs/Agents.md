#   Agents Folder
## Config Requirements
Although Agents do vary, your [Agent] section in your config file will look quite similar. Suppose that I wanted to create a DQN agent. I would set up the config the following way:
```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
Below you can see what config requirements different Agents have.

## Contents
*   Agent.py
*   DDPGAgent.py
*   DQNAgent.py
*   ImitationAgent.py
*   init_agent.py

##  Agent.py
___
Abstract Class
##  DDPGAgent.py
___
* An agent that can work with Basic Continous DDPG.
### Config Set Up     
```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
##  DQNAgent.py
___
* An agent for basic DQN.
### Config Set Up     
```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
##  ImitationAgent.py
___
* An agent for imitation learning.
### Config Set Up     
```
[Agent]
num_agents=2
optimizer_function='Adam'
learning_rate=0.03
action_policy='argmax'
```
##  init_agent.py
### Config Set Up     
___
* You have to add your modules here so that Shiva will see them.

##  ParameterizedDDPGAgent.py 
___
* An agent that can work Parameterized DDPG.
### Config Set Up     
```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
##  PPOAgent.py
### Config Set Up     
```
[Agent]
num_agents=1
optimizer_function='Adam'
learning_rate=0.0001
eps=1e-4
```
___
* An agent capable of running PPO.
### Config Set Up     

```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
##  ImitationAgent.py
___
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
   
### Config Set Up     
```
[Agent]
learning_rate=0.001
optimizer_function='Adam'
```
 
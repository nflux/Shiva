# Environments
___
## Contents
___
*   init
    * [Go to code]()
*   Environment
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/Environment.py)
*   GymContinuousEnvironment
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/GymContinuousEnvironment.py)
*   GymDiscreteEnvironment
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/GymDiscreteEnvironment.py)
*   RobocupDDPGEnvironment
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/RoboCupDDPGEnvironment.py)
*   UnityEnvironment
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/UnityEnvironment.py)


## init
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/__init__.py)
If you add environments, add them to this file as well.

##  Environment
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/Environment.py)
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

## Gym
___
Shiva supports both Gym Continuous Environments (e.g. MountainCarContinuous-v0) and Gym Discrete Environments (e.g. MountainCar-v0). The Discrete action space allows a fixed range of non-negative numbers while the Box action space represents an n-dimensional box.
###  GymContinuousEnvironment
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/GymContinuousEnvironment.py)
GymDiscreteEnvironment and GymContinuousEnvironment both inherit from the Environment class. The only difference between the two is that GymDiscreteEnvironment takes the argmax() of the action output from the network before passing it to the environment while GymContinuousEnvironment simply passes the action output.
#### Config Setup
```
[Environment]
type='GymContinuousEnvironment'
env_name='MountainCarContinuous-v0'
action_space='continuous'
observation_space="continuous"
render=True
num_agents=1

```

###  GymDiscreteEnvironment
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/GymDiscreteEnvironment.py)
GymDiscreteEnvironment and GymContinuousEnvironment both inherit from the Environment class. The only difference between the two is that GymDiscreteEnvironment takes the argmax() of the action output from the network before passing it to the environment while GymContinuousEnvironment simply passes the action output.
#### Config Setup
```
[Environment]
type='GymDiscreteEnvironment'
env_name='MountainCar-v0'
action_space='discrete'
observation_space="continuous"
render=True
num_agents=1
```

##  RobocupDDPGEnvironment
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/RoboCupDDPGEnvironment.py)
### Config Setup
```
[Environment]
type='RoboCupDDPGEnvironment'
env_name='RoboCup'
port=45000
; action level
action_level = 'low'
; feature level
feature_level = 'low'
env_render = False
rcss_log = False
hfo_log = False
num_ep = 100
ep_length = 500
untouched = 200
determ = True
burn_in = 500
record_lib = False
record_serv = False
num_left = 1
num_right = 0
num_l_bot = 0
num_r_bot = 0
left_bin = 'helios10'
right_bin = 'helios11'
goalie = False
; per episode
reward_anneal = 1_000_000
offense_ball = 0
sync_mode = True
fullstate = True
verbose = False
log = 'log'
; Ball position
ball_x_min = 0.0
ball_x_max = 0.0
ball_y_min = 0.0
ball_y_max = 0.0
; Agent Positions
agents_x_min=-0.50
agents_x_max=-0.50
agents_y_min=-0.10
agents_y_max=0.10
; Change Positions
change_ball_x=0.0
change_ball_y=0.0
change_agents_x=0.0
change_agents_y=0.0
change_every_x=100
init_env = True
```
## UnityEnvironment
___
[Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/UnityEnvironment.py)
### Config Setup
```
[Environment]
type='UnityEnvironment'
exec='shiva/envs/unitybuilds/3DBall/3DBall.x86_64'
env_name='3DBall'
action_space='continuous'
observation_space="continuous"
render=True
num_agents=1
```

# Learners
___

## Config Requirements
___
Are specified in detail below.

## Contents
___
* __init__
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/__init__.py)
* DDPGLearner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/DDPGLearner.py)
* Learner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/Learner.py)
* SingleAgentDDPGLearner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentDQNLearner.py)
* SingleAgentDQNLearner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentDQNLearner.py)
* SingleAgentImitationLearner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentImitationLearner.py)
* SingleAgentPPOLearner
    * [Go to code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentPPOLearner.py)


## __init__
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/__init__.py)


##  DDPGLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/DDPGLearner.py)

### Config Set Up     
```
[Learner]
type='DDPGLearner'
using_buffer=True
episodes=100000
load_agents=False
; load_agents = 'runs/ML-MountainCarContinuous-v0-10-31-20:00'
```

##  Learner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/Learner.py)

### Config Set Up     
```
[Learner]
type='DDPGLearner'
using_buffer=True
episodes=100000
load_agents=False
; load_agents = 'runs/ML-MountainCarContinuous-v0-10-31-20:00'
```

##  SingleAgentDDPGLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/DDPGLearner.py)

### Config Set Up     
```
[Learner]
type='SingleAgentDDPGLearner'
using_buffer=True
episodes=150
save_checkpoint_episodes=25
load_agents=False
; load_agents = 'runs/ML-MountainCarContinuous-v0-10-31-20:00'
manual_play = False
```

##  SingleAgentDQNLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentDQNLearner.py)

### Config Set Up     
```
[Learner]
type='SingleAgentDQNLearner'
using_buffer=True
episodes=10_000
save_frequency=500
;metrics =  ["Reward", "LossPerStep", "TotalReward"]
;load_path='runs/ML-CartPole-v0-10-24-02:06/L-0'
load_path=False
```


##  SingleAgentImitationLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentImitationLearner.py)

### Config Set Up     
```
[Learner]
type='SingleAgentImitationLearner'
using_buffer=True
supervised_episodes=10
imitation_episodes=10
dagger_iterations=2
save_frequency=5000
expert_agent='runs/ML-RoboCup-11-04-12:53/L-0'
```


##  SingleAgentPPOLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/learners/SingleAgentPPOLearner.py)

### Config Set Up     
```
[Learner]
type='SingleAgentPPOLearner'
using_buffer=True
episodes=500
load_agents=False
; load_agents = 'runs/ML-MountainCarContinuous-v0-10-31-20:00'
```
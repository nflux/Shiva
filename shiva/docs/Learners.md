# Learners
___

## Config Requirements
___
Are specified in detail below.

## Contents
___
* DDPGLearner
    * [Go to code](../shiva/learners/DDPGLearner.py)
* Learner
    * [Go to code](../shiva/learners/Learner.py)
* SingleAgentDDPGLearner
    * [Go to code](../archive/SingleAgentDQNLearner.py)
* SingleAgentDQNLearner
    * [Go to code](../archive/SingleAgentDQNLearner.py)
* SingleAgentImitationLearner
    * [Go to code](../archive/SingleAgentImitationLearner.py)
* SingleAgentPPOLearner
    * [Go to code](../archive/SingleAgentPPOLearner.py)

##  DDPGLearner
___
[Go to Code](../shiva/learners/DDPGLearner.py)

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
[Go to Code](../shiva/learners/Learner.py)

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
[Go to Code](../shiva/learners/DDPGLearner.py)

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
[Go to Code](../archive/SingleAgentDQNLearner.py)

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
[Go to Code](../archive/SingleAgentImitationLearner.py)

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
[Go to Code](../archive/SingleAgentPPOLearner.py)

### Config Set Up     
```
[Learner]
type='SingleAgentPPOLearner'
using_buffer=True
episodes=500
load_agents=False
; load_agents = 'runs/ML-MountainCarContinuous-v0-10-31-20:00'
```
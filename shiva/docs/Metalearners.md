# Metalearners

# Config Requirements

# Contents
*   __init__ 
    *   [Link to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/__init__.py)
*   MetaLearner
    *   [Link to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/MetaLearner.py)
*   MultipleAgentMetaLearner
    *   [Link to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/MultipleAgentMetaLearner.py)
*   SingleAgentMetaLearner
    *   [Link to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/SingleAgentMetaLearner.py)

## __init__
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/__init__.py)

Add any metalearner modules you add to Shiva to this file so Shiva can see it.

## MetaLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/MetaLearner.py)

Abstract class all MetaLearners inherit from.

## MultipleAgentMetaLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/MultipleAgentMetaLearner.py)
### Config Set Up     
```
[MetaLearner]
type='MultipleAgentMetaLearner'
start_mode="production"
; turn in to list of learners later
learner_list= 5
learning_rate= [0.0, 0.05]
optimize_env_hp= False
optimize_learner_hp= False
evolution= False
updates_per_iteration = 50
exploit = 't_Test'
explore = 'perturbation'
p_value = 0.05
perturbation_factors = [0.8,1.2]
```

## SingleAgentMetaLearner
___
[Go to Code](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/metalearners/SingleAgentMetaLearner.py)
### Config Set Up     
```
[MetaLearner]
type='SingleAgentMetaLearner'
start_mode="production"
optimize_env_hp=False
optimize_learner_hp=False
evolution=False
```
# Metalearners

# Config Requirements

# Contents
*   __init__ 
    *   [Link to Code](../shiva/metalearners/__init__.py)
*   MetaLearner
    *   [Link to Code](../shiva/metalearners/MetaLearner.py)
*   MultipleAgentMetaLearner
    *   [Link to Code](../shiva/metalearners/MultipleAgentMetaLearner.py)
*   SingleAgentMetaLearner
    *   [Link to Code](../shiva/metalearners/SingleAgentMetaLearner.py)

## MetaLearner
___
[Go to Code](../shiva/metalearners/MetaLearner.py)

Abstract class all MetaLearners inherit from.

## MultipleAgentMetaLearner
___
[Go to Code](../shiva/metalearners/MultipleAgentMetaLearner.py)
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
[Go to Code](../shiva/metalearners/SingleAgentMetaLearner.py)
### Config Set Up     
```
[MetaLearner]
type='SingleAgentMetaLearner'
start_mode="production"
optimize_env_hp=False
optimize_learner_hp=False
evolution=False
```
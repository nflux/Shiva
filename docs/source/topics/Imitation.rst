==================
Imitation Learning
==================

Imitation Learning is a learning technique which aims to mimic expert behavior in order to increase the 
initial training speed and efficiency of a new agent. Our current implementation has two phases and
can be found `here <https://github.com/nflux/Control-Tasks/blob/docs/shiva/shiva/algorithms/ImitationAlgorithm.py>`_.

.. rubric:: 1. Supervised Phase

During the supervised learning phase, the new agent learns on prerecorded episodes from the expert. 
While this Behavior Cloning technique is useful, it has limitations. The major limitation being that 
the new agent is never exposed to dangerous states/observations, because the expert agent knows how to 
avoid such situations. For example, an expert driver is never going to end up with their car facing a 
wall, because a true expert driver knows that there is no value in steering into that position 
(aside from parking that is). To build on the value created from the supervised policy learning we move 
on to the second phase.

.. rubric:: 2. Dagger Algorithm Phase

This phase is an iterative process that increases the new agents exposure to new, and potentially dangerous, 
states/observations. In this portion we let the new agent control the trajectories of the episodes and then 
correct it's behavior with the expert agent. This is like when a teenager is learning to drive for the first 
time. The teenager is in control of the car, but either a parent or driving instructor is there to correct 
bad behavior displayed by the new driver. 

Dagger stands for Data Aggregation, and it works by aggregating all of the data (episodes) traversed during 
the learning process. Initially this is the prerecorded expert episodes, but at each new iteration the episodes 
lead by the new agent policies are added to the dataset. At the start of each iteration a new policy is trained 
on all of the collected data. At it's core Dagger is supervised training a new policy with episodes created by 
an expert and previous learning iteration policies. While the technique is simple, it is a very powerful and 
effective algorithm. The main downside of the dagger algorithm is that it requires access to the expert during 
the training period.
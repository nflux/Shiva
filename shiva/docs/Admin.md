# Shiva Admin

## Contents
*   [ShivaAdmin.py](../shiva/core/ShivaAdmin.py).

## Overview

The ShivaAdmin class handles and simplifies the file management and administrative tasks for the project such as
* Track and create file directories for the [MetaLearner](../shiva/metalearners), [Learner](../shiva/learners), [Agent](../shiva/agents)
* Handle the saving and loading of
    * config files
    * class pickles
    * networks
* Save metrics for Tensorboard visualizations

## Usage

Requires the following section in the config file

```
[Admin]
save =              True
traceback =         True
directory =         {'runs': '/runs'}
```

And it’s accessible with one simple import

```python
from shiva.core.admin import Admin
```
## Saving

The agents will be saved in the [runs](../runs) under their corresponding MetaLearner and Learner folder. The config used, the Learner and Agents classes will be saved with their corresponding networks and parameters.

From the Learner class, just do a

```python
Admin.update_agents_profile(self)
```

**Note**
  - self is the Learner class
  - Make sure the MetaLearner have added their profiles with Admin before any saving. A common workflow of the MetaLearner would be:
  
```python
self.learner = self.create_learner()
Admin.add_learner_profile(self.learner)
self.learner.launch() # learner launches a whole learning instance
Admin.update_agents_profile(self.learner)
self.save()
```

## TensorBoard

To save metrics on Tensorboard, use the following Admin functions
```python
def init_summary_writer(self, learner, agent) -> None:
        '''
            Instantiates the SummaryWriter for the given agent
            Input
                @learner            Learner instance owner of the Agent
                @agent              Agent who we want to records the metrics
        '''
```

```python
    def add_summary_writer(self, learner, agent, scalar_name, value_y, value_x) -> None:
        '''
            Adds a metric to the tensorboard of the given agent
            Input
                @learner            Learner instance owner of the agent
                @agent              Agent who we want to add
                @scalar_name        Metric name
                @value_y            Usually the metric
                @value_x            Usually time
        '''
```

Do a simple call from the learners such as

```python
Admin.add_summary_writer(self, self.agent, 'Total_Reward_per_Episode', self.totalReward, self.ep_count)
```

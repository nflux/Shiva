# ShivaAdmin

## Overview

The ShivaAdmin class handles the file management and administrative tasks for the project such as
* Track and create file directories for the [MetaLearner](https://github.com/nflux/Control-Tasks/tree/dev/shiva/shiva/metalearners), [Learner](https://github.com/nflux/Control-Tasks/tree/dev/shiva/shiva/learners), [Agent](https://github.com/nflux/Control-Tasks/tree/dev/shiva/shiva/agents)
* Handle the saving and loading of
    * config files
    * class pickles
    * networks
* Save metrics for Tensorboard visualizations

It’s accessible with one simple import

```python
from settings import shiva
```

And requires the following section in the config file

```
[Admin]
save =              True
traceback =         True
directory = {'runs': '/runs'}
```

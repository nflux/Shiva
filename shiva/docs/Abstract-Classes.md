# Abstract Classes

What makes Shiva a great tool is its flexibility and adaptability.  Shiva is based around seven abstract classes from which every component in a model/pipeline is based off of.

Currently we have the following abstract classes:
```
* MetaLearner
    [More Info](Metalearners.md)
* Learner Abstract
* Algorithm Abstract 
* Environment Abstract
* ReplayBuffer Abstract 
* Agent Abstract 
* Network Abstract
```
They are all templates of reinforcement learning objects that work together to build a model. They are like interfaces in that they define what parameters and functions are expected from an objects that inherit from them. Every component

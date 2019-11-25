# Abstract Classes

What makes Shiva a great tool is its flexibility and adaptability.  Shiva is based around seven abstract classes from which every component in a model/pipeline is based off of.

Currently we have the following abstract classes:
```
* MetaLearner
    [More Info](Metalearners.md)
* Learner Abstract
    [More Info](Learners.md)
* Algorithm Abstract 
    [More Info](Algorithms.md)
* Environment Abstract
    [More Info](Environments.md)
* ReplayBuffer Abstract 
    [More Info](Buffers.md)
* Agent Abstract 
    [More Info](Agents.md)
* Network Abstract
    [More Info](Networks.md)
```

Following this pattern makes your components modular enough to be reusable and easily debugged. You can look at them like they are all templates of reinforcement learning objects that work together to build a model. They are like interfaces in that they define what parameters and functions are expected from an objects that inherit from them. You are able to inherit from them and immediately your components are connected to Shiva. You can click on the links to learn more about the individual components and see what Shiva already as implemented.

The foundation of Shiva is the abstract classes from which every component of a model is based off of.

Currently we have the following abstract classes:

AbstractMetaLearner, AbstractLearner, AbstractAlgorithm, AbstractEnvironment, AbstractReplayBuffer, AbstractAgent, AbstractNetwork.

They are all templates of reinforcement learning objects that work together to build a model. They are like interfaces in that they define what parameters and functions are expected from an objects that inherit from them. 

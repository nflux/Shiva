# The Abstracts 
* The abstracts are our super classes that contains the templates and functions that the child classes are expected to follow. Each of these abstract templates grabs the config's from their assigned portion from the .ini file, which gives the user a more refined settings to suit their needs. 
## The Abstracts Includes
 * AbstractMetaLearner
 * AbstractLearner
 * AbstractAlgorithm
 * AbstractEnvironment
 * AbstractReplayBuffer
 * AbstractAgent
### AbstractMetaLearner
 * The abstractMetalearner or [Metalearner.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/metalearners/MetaLearner.py) contains the templete functions to start the entire learning sequence of the agent, which contains functions like exploit explore, genetic crossover, evolve, evaluate, and recordmetrics, create learners, creates learner id , and also the save function. It also grabs the functions decided from the config files. 
### AbstractLearner
 * The abstractLearner or [Learner.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/learners/Learner.py) contains the templete functions for the learning functions required to learn. This includes grathering the configs for learner. The functions include update, steps, create environment, getting the agents. getting the algorithm, a save function, a launch function to start the process of creating the agent and learning, a load function, to load agents, and a get id function to help the the agent ID. 
### AbstractAlgorithm
 *  The AbstractLearner or [Algorithm.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/algorithms/Algorithm.py) contains the template that all of our algorithms that follows. When initalized the Abstract Algorithm grabs the config settings, creates an agent count, intiates a list of agents, creates an attribute of observation/action space, loss calcuation and enables gpu, using CUDA when available. 
 * Additional functions includes updates (updates the agents network using the data), get_action (Determines the best action for the agent from the observation), create_agent (creates a new agent)), id_generator (creates an ID for the agent), get_agents (grabs the agents).
### AbstractEnvironment
 *  The AbstractEnvironment or [Environment.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/envs/Environment.py) is the template that allows us to grab the attributes from the environment. This includes: step, observations, actions, rewards, observation space, action space, current step, reset, load viewer, and normalized rewards. 
### AbstractReplayBuffer
 *  The AbstractReplayBuffer or [ReplayBuffer.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/buffers/ReplayBuffer.py) contains the template in, which how the observations, rewards, actions, and agents are stored in an agent's recent actions. 
### AbstractAgent
 * The AbstractAgent or [Agent.py](https://github.com/nflux/Control-Tasks/blob/unity/shiva/shiva/agents/Agent.py) The AbstracAgent contains templates that creates agent attributes like id, observation space, action space, agent configuration, and network configuration.
 * The template also contains the save function, loading network function as well as getting the best action functions

# Abstract Classes
What makes Shiva a great tool is its flexibility and adaptability.  Shiva is based around seven abstract classes from which every component in a model/pipeline is based off of.

The abstracts are our super classes that contains the templates and functions that the child classes are expected to follow. Each of these abstract templates grabs the config's from their assigned portion from the .ini file, which gives the user a more refined settings to suit their needs. 

Following this pattern makes your components modular enough to be reusable and easily debugged. You can look at them like they are all templates of reinforcement learning objects that work together to build a model. They are like interfaces in that they define what parameters and functions are expected from an objects that inherit from them. You are able to inherit from them and immediately your components are connected to Shiva. You can click on the links to learn more about the individual components and see what Shiva already as implemented.

### [AbstractMetaLearner](../shiva/learners/MetaLearner.py)
 * The AbstractMetalearner contains the template functions to start the entire learning sequence of the agent, which contains functions like exploit explore, genetic crossover, evolve, evaluate, and record metrics, create learners, creates learner id , and also the save function. It also grabs the functions decided from the config files. 
### [AbstractLearner](../shiva/learners/Learner.py)
 * The AbstractLearner contains the template functions for the learning functions required to learn. This includes gathering the configs for learner. The functions include update, steps, create environment, getting the agents. getting the algorithm, a save function, a launch function to start the process of creating the agent and learning, a load function, to load agents, and a get id function to help the the agent ID. 
### [AbstractAlgorithm](../shiva/algorithms/Algorithm.py)
 *  The AbstractAlgorithm contains the template that all of our algorithms follows. When initialized the Abstract Algorithm grabs the config settings, creates an agent count, initiates a list of agents, creates an attribute of observation/action space, loss calculation and enables gpu, using CUDA when available. 
 * Additional functions includes updates (updates the agents network using the data), get_action (Determines the best action for the agent from the observation), create_agent (creates a new agent)), id_generator (creates an ID for the agent), get_agents (grabs the agents).
### [AbstractEnvironment](../shiva/envs/Environment.py)
 *  The AbstractEnvironment is the template that allows us to grab the attributes from the environment. This includes: step, observations, actions, rewards, observation space, action space, current step, reset, load viewer, and normalized rewards. 
### [AbstractReplayBuffer](../shiva/buffers/ReplayBuffer.py)
 *  The AbstractReplayBuffer contains the template in, which how the observations, rewards, actions, and agents are stored in an agent's recent actions. 
### [AbstractAgent](../shiva/agents/Agent.py)
 * The AbstractAgent contains templates that creates agent attributes like id, observation space, action space, agent configuration, and network configuration.
 * The template also contains the save function, loading network function as well as getting the best action functions
### [AbstractNetwork](../shiva/networks/)
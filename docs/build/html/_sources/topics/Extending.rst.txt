===============
Extending Shiva
===============

.. rubric:: Implementing a new Algorithm in Shiva

Shiva is a work in progress and collaborators are encouraged to implement their 
algorithms in Shiva's framework. If you want to implement something new you can 
see what components of Shiva are reusable and what is needed to accomplish your 
goal. If, for example, you wanted to implement Soft Actor Critic, then there are 
many modules that you can reuse. Typically you need an Agent, Algorithm, Learner, 
MetaLearner, Replay Buffer, and an Environment. If its a Unity
environment then you can use our UnityWrapperEnvironment, you may also reuse the 
replay buffer, and in the case of a single agent, you can use SingleAgentMetaLearner.py. 
The things you may need to implement are the Algorithm, Agent, and Learner (if the 
training loop is drastically different from the currently available learners) classes. 
(We are planning on refactoring the code soon to become more generalized so that 
learners can be shared across various implementations).

Now, you might create a SoftActorCriticAlgorithm.py file in the algorithms folder. Then you
would import SoftActorCriticAlgorithm.py inside of MPILearner. When you implement SAC you would
to develop an SACAgent that adhere's to MPILearner's structure. A very good reference would
be MADDPGAgent.py because it is our most advanced agent that utilizes all of MPILearner's features.
If you are using a Unity environment, you need to place your binary file and data inside of the
unitybuilds/ directory inside of the environments module. As you are building the algorithm and agent,
you can add whatever configurable attributes you want/need to that version of the learner, agent,
and algorithm to have inside of the configuration file. You might call the correspodning configuration
files SAC/Unity-3DBall.ini and SAC/Agent-3DBall.ini. There are a lot of config files available for
reference and we can add more instructions upon request.
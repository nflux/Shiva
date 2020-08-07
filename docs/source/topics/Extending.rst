# Implementing a new Algorithm in Shiva

Shiva is a work in progress and collaborators are encouraged to implement their algorithms in Shiva's framework. If you want to implement 
something new you can see what components of Shiva are reusable and what is needed to accomplish your goal. If, for example, you wanted to 
implement Soft Actor Critic, then there are many modules that you can reuse. Typically you need an Agent, Algorithm, Learner, MetaLearner, 
Replay Buffer (not always), and an Environment. If its a Unity environment then you can use our UnityWrapperEnvironment, you may also reuse 
the replay buffer, and in the case of a single agent, you can use SingleAgentMetaLearner.py. The things you may need to implement are the 
Algorithm, Agent, and Learner (if the training loop is drastically different from the currently available learners) classes. (We are planning 
on refactoring the code soon to become more generalized so that learners can be shared across various implementations).

Now, you might create a SoftActorCritic.py file in the algorithms folder. Then you would import SoftActorCritic.py inside of the __init__.py 
folder inside of the algorithms module. Implementing SAC in Shiva may or may not require an SACLearner. If it does you’d have to add the corresponding 
SACLearner inside of the learner module and import it in the corresponding _init_.py file in the learner module. If you are using a Unity environment, 
you need to place your binary file and data inside of the unitybuilds/ directory inside of the environments module. As you are building the algorithm 
and learner, you can add whatever configurable attributes you want/need to that version of the learner or algorithm to have inside of the configuration 
file. You might call the configuration file SAC-3DBall ini. There are a lot of config files available for reference and we can add more instructions 
upon request.

~~# Implementing a new Algorithm in Shiva (old?)~~

~~Shiva is pretty great and it may have everything you need or it may not. If you want to implement something new you have to see what components 
of Shiva are reusable and what you actually need to add to accomplish your goal. If, for example, you wanted to implement Soft Actor Critic, then 
there are probably a lot of stuff you can reuse. Typically you need an Agent, Algorithm, Learner, MetaLearner, Replay Buffer (not always), and an 
Environment. If its a Unity environment then you can use our UnityWrapperEnvironment, you can also probably reuse our replay buffer, and in the case 
of a single agent, you can use SingleAgentMetaLearner.py. The only things you might need to implement are the Algorithm, Agent, and possibly a Learner 
if it runs drastically different from other learners. (We are planning on refactoring the code soon to become more generalized so its possible that 
in the future more learners might be shared across various implementations).~~

~~Now, you might create a SoftActorCritic.py file in algorithms folder. Then you’d have import SoftActorCritic.py inside of the __init__.py folder 
inside of the algorithms module. Implementing SAC in Shiva may or may not require an SACLearner. If it does you’d have to add the corresponding 
SACLearner inside of the learner module and import it in the corresponding _init_.py file in the learner module. If you want to implement it for a 
single agent or multi agents will also be at your discretion if you are to implement something new Shiva doesn’t have. If you are using a Unity 
environment, all you need to do is place your binary file and data inside of the unitybuilds/ directory inside of the environments module. As you are 
building the algorithm and learner, you can add whatever configurable attributes you want/need that version of the learner or algorithm to have inside 
of the configuration file. You might call the configuration file SAC-3DBall.ini. There are a lot of config files available for reference but if more 
documentation is necessary or requested we will do what we can to make things more clear.~~
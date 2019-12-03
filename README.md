# Shiva

Deep reinforcement/imitation learning pipeline leveraging population based training for continuous and discrete action spaces. Adaptable to game environments by creating a game specific environment wrapper.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

# Requirements
Must have Ubuntu 16.04
Python 3.5+

# Installing Shiva

In order to install Shiva, you have to get all parts of Shiva functional. There is a bash installation script immediately inside of the repo called install.sh. Running it will install Shiva for you given that you have Ubuntu 16.04. If, for some reason, the script fails on Ubuntu 16.04 open up an issue and we'll look into it but you can also more or less follow the instructions below and build it from source yourself.

## Building Shiva

To build Shiva from source we first clone the repo.

```
git clone https://github.com/nflux/Control-Tasks/
```

Then we can go into the project structure and start setting up components and dependencies. 

### Robocup

One of the biggest components inside of Shiva is Robocup, an environment excellent for running multi agent algorithms.
```
cd Control-Tasks/shiva/shiva/envs/robocup
git clone https://github.com/nflux/rcssserver.git
git clone https://github.com/nflux/librcsc.git
```
Whether you’re using Ubuntu, Manjaro, or MacOS, you need to install cmake, C++ Boost libraries, Flex, and Qt version 4.

### Installing RoboCup in Ubuntu

If on Ubuntu you can use apt-get package manager.
```
sudo apt-get install cmake libboost-dev libboost-all-dev flex qt4-default
```
### Installing RoboCup in Manjaro

In Manjaro you can use pacman package manager.
```
sudo pacman -S cmake boost flex qt4
```
### Installing RoboCup MacOS

In MacOS you can use homebrew?
```

```
Now that you’ve installed the libraries you should be able to build robocup servers. If you are still inside of the robocup folder then simple type the following, otherwise navigate back to the robocup folder and then proceed.
```
cd HFO
bash compile.sh
```

### Installing Unity API
Now, you want to set up your python virtual environment. You have plenty of options to choose from (anaconda, venv, virtualenv, etc) but regardless which virtual environment manager you use there is a requirements.txt file that you can use to quickly install all the python dependencies. To install the Unity API you have to go to Control-Tasks/shiva/shiva/envs/ml-agents/ml-agents and run

```
cd Control-Tasks/shiva/shiva/envs/ml-agents/ml-agents-envs/
pip install -e ./
cd ..
cd ml-agents/
pip install -e ./
```
Finally, in case anything was missed do pip install requirements to make sure you have the rest of the dependencies.
```
pip install requirements.txt
```
At this point you should have shiva installed and ready to run but it would behoove you to make sure you have cuda properly installed because Shiva detects your GPU and can greatly speed up training.

## Some Notes on Unity

This version of Shiva uses a gym wrapper around Unity. This version uses a Unity API to pass the actions and collect observations from the Unity environment. You only need to place your Unity binary and data files inside of unitybuilds/ directory immediately inside Shiva's environment module. More instructions can be found here [Unity Documentation](https://github.com/nflux/Control-Tasks/blob/master/shiva/docs/Unity.md) 


# Quick Start

Now with Shiva installed you are ready to see what Shiva can do. Let’s use an algorithm already implemented in Shiva and deploy inside of OpenAI gym environment. Simply run the following commands:
```
cd Control-Tasks/shiva
python -W ignore shiva -config DQN.ini -name testRun 
```
This will run a DQN algorithm inside of a gym environment to solve the classical reinforcement learning problem Cartpole.

You could also try the following command to see DDPG solve Mountain Car Continuous.
```
python -W ignore shiva -c ContinuousDDPG.ini -n testRun
```
These gym environments can be used as sanity checks for algorithms in very complex environments to make sure that the algorithm is or isn’t the problem.

You can also have sanity checks with really simple environments like Basic and 3DBall Unity Environments. If you want to run either of those run the following commands:

```
python -W ignore shiva -c DDPG-3DBall.ini -n run
```
This will run Continuous DDPG with the 3D Ball environment.
or 
```
python -W ignore shiva -c Unity.ini -n run
```
This will run DQN in the Basic environment.

There are a lot of configuration files [here](https://github.com/nflux/Control-Tasks/tree/master/shiva/configs) that you can reference as well as the documentation found [here](https://github.com/nflux/Control-Tasks/tree/master/shiva/docs). Also every module has config examples in their respective readme.

# Implementing a new Algorithm in Shiva

Shiva is a work in progress and collaborators are encouraged to implement their algorithms Shiva's framework. If you want to implement something new you can see what components of Shiva are reusable and what you actually need to add to accomplish your goal. If, for example, you wanted to implement Soft Actor Critic, then there are many modules that you can reuse. Typically you need an Agent, Algorithm, Learner, MetaLearner, Replay Buffer (not always), and an Environment. If its a Unity environment then you can use our UnityWrapperEnvironment, you may also reuse the replay buffer, and in the case of a single agent, you can use SingleAgentMetaLearner.py. The things you may need to implement are the Algorithm, Agent, and Learner (if the training loop is drastically different from the currently available learners) classes. (We are planning on refactoring the code soon to become more generalized so that learners can be shared across various implementations). 

Now, you might create a SoftActorCritic.py file in the algorithms folder. Then you would import SoftActorCritic.py inside of the __init__.py folder inside of the algorithms module. Implementing SAC in Shiva may or may not require an SACLearner. If it does you’d have to add the corresponding SACLearner inside of the learner module and import it in the corresponding _init_.py file in the learner module. If you are using a Unity environment, you need to place your binary file and data inside of the unitybuilds/ directory inside of the environments module. As you are building the algorithm and learner, you can add whatever configurable attributes you want/need to that version of the learner or algorithm to have inside of the configuration file. You might call the configuration file SAC-3DBall ini. There are a lot of config files available for reference and we can add more instructions upon request.

# Restrictions

If you would like to contribute to Shiva, we would like you to do so by providing your own implementations of the abstract modules to maintain stability. If you have difficulties with any of the existing modules please raise an issue on the repository.

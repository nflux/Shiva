# Quick Start

You want to see what Shiva can do? Let’s use an algorithm already implemented in Shiva and deploy inside of OpenAI gym environment. Simply run the following commands:

> cd Control-Tasks/shiva
> python -W ignore shiva -config DQN.ini -name testRun 

This will run a DQN algorithm inside of a gym environment to solve the classical reinforcement learning problem Cartpole.

You could also try the following command to see DDPG solve Mountain Car Continuous.

> python -W ignore shiva -c ContinuousDDPG.ini -n testRun

These gym environments can be used as sanity checks for algorithms in very complex environments to make sure that the algorithm is or isn’t the problem.

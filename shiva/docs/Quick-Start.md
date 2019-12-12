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
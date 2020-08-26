# Quick Start

Now with Shiva installed you are ready to see what Shiva can do. 

Let’s use an algorithm already implemented in Shiva and deploy inside of OpenAI Gym environment. Simply run the following commands:

```bash
cd Control-Tasks/shiva
python shiva -c MADDPG/Gym-Cartpole.ini
```

This will run a MADDPG algorithm inside of a Gym environment to solve the classical reinforcement learning problem Cartpole.

You could also try the following command to see it solve Mountain Car Continuous.
```
python shiva -c MADDPG/Gym-MountainCarContinuous.ini
```

These Gym environments can be used as sanity checks for algorithms in very complex environments to make sure that the algorithm is or isn’t the problem.

You can also have sanity checks with really simple environments like Basic and 3DBall Unity Environments. If you want to run either of those run the following commands:

Note that the Unity binaries are necessary!

```
python shiva -c MADDPG/Unity-3DBall.ini
```

There are a lot of configuration files [here](../configs) that you can reference as well as the documentation found [here](./Config-Files.md). Also every module has config examples in their respective readme.

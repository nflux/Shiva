Quick Start
===========

Now with Shiva installed you are ready to see what Shiva can do. 

Let’s use an algorithm already implemented in Shiva and deploy inside of OpenAI Gym environment. Simply run the following commands:

.. code-block:: python

   cd Control-Tasks/shiva
   python main.py -c DQN.ini

This will run a DQN algorithm inside of a Gym environment to solve the classical reinforcement learning problem Cartpole.

You could also try the following command to see DDPG solve Mountain Car Continuous.

.. code-block:: python

   python main.py -c ContinuousDDPG.ini

These Gym environments can be used as sanity checks for algorithms in very complex environments to make sure that the algorithm is or isn’t the problem.

You can also have sanity checks with really simple environments like Basic and 3DBall Unity Environments. If you want to run either of those run the following commands:

.. code-block:: python

   python main.py -c DDPG-3DBall.ini

This will run Continuous DDPG with the 3D Ball environment.

or 

.. code-block:: python

   python main.py -c Unity.ini

This will run DQN in the Unity Basic environment.

There are a lot of configuration files inside Control-Tasks/shiva/configs that can be referenced.

Quick Start
===========

Now with Shiva installed you are ready to see what Shiva can do. Go to the following directory:

.. code-block:: bash

   Control-Tasks/shiva


Let’s use an algorithm already implemented in Shiva and deploy inside of OpenAI Gym environment. Simply run the following commands:

.. code-block:: python

   python3 shiva -c MADDPG/Gym-CartPole.ini

This will run an MADDPG algorithm inside of a Gym environment to solve the classical reinforcement learning problem Cartpole.

These Gym environments can be used as sanity checks for algorithms in very complex environments to make sure that the algorithm isn’t the problem.

Unity's 3DBall Environment can also be a nice sanity check. If you want to run 3DBall try:

.. code-block:: python

   python3 shiva -c MADDPG/Unity-3DBall.ini

This will run Continuous MADDPG.


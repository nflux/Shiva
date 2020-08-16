Welcome to Shiva's Documentation!
=================================

Shiva is built to be simulation-engine agnostic, its framework is abstracted to support various types of observation and
actions spaces with different environment settings and number of agents. Additionally, Shiva is designed to support
distributed processing across a large number of servers to support learning in complex environments with large
observation or action spaces where multiple agents need to converge to a team policy. At the moment, Shiva supports
popular reinforcement and imitation learning algorithms such as Deep Q-Network (DQN), Deep Deterministic Policy Gradient
(DDPG), Proximal Policy Optimizations (PPO), Multi Agent Deep Deterministic Policy Gradient (MADDPG), Dataset
Aggregation (DAGGER) method in addition to a few customized and hybrid model-based algorithms that leverage the dynamics
of the environment to converge to at a faster rate. The framework is built to enable researchers to design and
experiment with new algorithms and be able to test them at scale in different environments and scenarios with minimum
setup on the infrastructure.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

First Steps
===========
Are you new to Shiva? This is the place to start!

* **From scratch:**
  :doc:`Installation <intro/Install>`

How the Documentation is Organized
==================================

We plan on expanding Shiva's documentation over time which will probably lead
to a lot of documentation. A high-level overview of how it's organized
will help you know where to look for certain things:

* :doc:`Tutorials </intro/index>` take you by the hand through a series of
  steps to use Shiva. Start here if you're new to Shiva or Reinforcement
  Learning.

* :doc:`Topic guides </topics/index>` discuss key topics and concepts at a
  fairly high level and provide useful background information and explanation.

* :doc:`How-to guides </howto/index>` are recipes. They guide you through the
  steps involved in addressing key problems and use-cases. They are more
  advanced than tutorials and assume some knowledge of how Shiva works.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

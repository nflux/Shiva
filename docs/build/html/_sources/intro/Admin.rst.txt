===========
Shiva Admin
===========

.. rubric:: Contents
*   Shiva Admin source code can be found `here. <https://github.com/nflux/Control-Tasks/blob/docs/shiva/shiva/core/ShivaAdmin.py>`_

.. rubric:: Overview

The ShivaAdmin class handles and simplifies the file management and administrative tasks for the project such as

* Track and create file directories for the `MetaLearner and Learners <https://github.com/nflux/Control-Tasks/tree/docs/shiva/shiva/learners>`_, `Agent <https://github.com/nflux/Control-Tasks/tree/docs/shiva/shiva/agents>`_.

* Handle the saving and loading of
    * config files
    * class pickles
    * networks

* Save metrics for Tensorboard visualizations

.. rubric:: Usage

Requires the following section in the config file

.. code-block:: ini

   [Admin]
   iohandler_address   = 'localhost:50001'
   print_debug         = True
   save                = True
   traceback           = True
   directory           = {'runs': '/runs/Gym-Cartpole/'}
   profiler            = True
   time_sleep = {'MetaLearner':    0,
                 'MultiEnv':        0.01,
                 'EvalWrapper':     1,
                 'Evaluation':      0.1}

   ; verbose levels for logs and terminal output
   ;   0 deactivated
   ;   1 debug
   ;   2 info
   ;   3 details
   log_verbosity = {
       'Admin':        0,
       'IOHandler':    1,
       'MetaLearner':  1,
       'Learner':      3,
       'Agent':        0,
       'Algorithm':    0,
       'MultiEnv':     1,
       'Env':          1,
       'EvalWrapper':  1,
       'Evaluation':   3,
       'EvalEnv':      0
       }


And it’s accessible with one simple import

.. code-block:: python

   from shiva.core.admin import Admin

.. rubric:: Saving

The agents will be saved in the `runs <https://github.com/nflux/Control-Tasks/tree/docs/shiva/runs>`_ directory inside their corresponding 
MetaLearner and Learner folder. The config used, the Learner and Agents classes will be saved with their corresponding networks 
and parameters.

From the Learner class, just do a

.. code-block:: python

   Admin.update_agents_profile(self)


.. note:: Note
  - self is the Learner class
  - Make sure the MetaLearner have added their profiles with Admin before any saving. A common workflow of the MetaLearner would be:
  
.. code-block:: python

   self.learner = self.create_learner()
   Admin.add_learner_profile(self.learner)
   self.learner.launch() # learner launches a whole learning instance
   Admin.update_agents_profile(self.learner)
   self.save()

.. rubric:: TensorBoard

To save metrics on Tensorboard, use the following Admin functions

.. code-block:: python

   def init_summary_writer(self, learner, agent) -> None:
        """ Instantiates the SummaryWriter for the given agent

            Args:
                learner:            Learner instance owner of the Agent
                agent:              Agent who we want to records the metrics

            Returns:
                None
        """

.. code-block:: python

   def add_summary_writer(self, learner, agent, scalar_name, value_y, value_x) -> None:
        """ Adds a metric to the tensorboard of the given agent

            Args:
                learner:            Learner instance owner of the agent
                agent:              Agent who we want to add
                scalar_name:        Metric name
                value_y:            Usually the metric
                value_x:            Usually time

            Returns:
                None
        """

Do a simple call from the learners such as

.. code-block:: python

   Admin.add_summary_writer(self, self.agent, 'Total_Reward_per_Episode', self.totalReward, self.ep_count)

========================
Running Gym Environments
========================

Gym Environments were the first environments we used to develop Shiva and often serve as sanity checks when we are testing
new algorithms. The config guide, found :doc:`here <Configs>`, uses the Gym-Cartpole.ini file as the basis for the guide but we'll go over some
details that are glossed over there for sake of staying on topic.


.. rubric:: OpenAI Gym Environments

Shiva supports Box2D and Classic Control Gym Environments. As long as you have installed the environments following the instructions
found in OpenAI's Gym Repo `here <https://github.com/openai/gym#installation>`_.


.. rubric:: Custom Gym Environments

If you happen to have created your own Gym Environment, here are some instructions on how you would hook it up with Shiva,
after a conversation with a user who decided to do just this we implemented a simple way to interface with Shiva given that
your environment file is properly structured.

`Here <https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa>`_ is an guide that details how you would 
set up your custom environment.

Once you've given your custom environment the proper architecture, registered, and pip installed it you'll be able to use it with Shiva
by setting up the Environment section of your ini file as follows. The following example is using the environment from the article as the
example.

.. note::

   One thing to watch for that's not mentioned in the article is that if you don't end the environment name with -v[0-9] it will give you
   an error

.. code-block:: ini

   [Environment]
   device               =   'gpu'
   type                 =   'GymEnvironment'
   env_name             =   'foo-v0'
   episode_max_length   =   100
   expert_reward_range  =   {'Custom-v0': [80, 100]}
   num_envs             =   1
   custom               =   'gym_foo'
   render               =   False
   port                 =   5010
   normalize            =   False
   reward_factor        =   0.1
   min_reward           =   0
   max_reward           =   1
   episodic_load_rate   =   1

In order to use your custom environemt it has be imported so in the custom hyperparameter type in the name of your custom gym environment module.
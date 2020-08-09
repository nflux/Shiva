Configuration Files
===================

We store all hyperparameters as well as data needed to run the models inside of two ini configuration files, the main and agent files. The naming
convention is that the main config is of the form 

Env-EnvName.ini 

and 

Agent-EnvName.ini

Shiva's architecture requires Metalearner, Learners, Algorithms, Agents, Networks, Environments, and Replay Buffers components to be able 
to run. In order to set up a run you need to configure each module in your pipeline. As such the configuration file requires a section for 
each component. We split up

Therefore your configuration file might look something like this:

.. rubric:: Test config for MADDPG Gym Cartpole

.. note::

    options within sections cannot have the same names


Main Config: Gym-CartPole.ini
=============================

.. rubric:: MetaLearner Section

The Metalearner oversees the learners and will be able to do population based tuning on the hyperparameters so how you want the Metalearner to run 
would be configured here.

You specify the type of MetaLearner, the mode (production or evaluation), whether there is evolution, and if we are optimizing the hyperparameters.

.. code-block:: ini

   [MetaLearner]
   type                         =   'MPIPBTMetaLearner'
   pbt                          =   False
   num_mevals                   =   1
   learners_map                 =   {'configs/MADDPG/Agent-Cartpole.ini': ['CartPole-v0']}
   num_learners_maps            =   1
   num_menvs_per_learner_map    =   1
   manual_seed                  =   4


.. rubric:: Evaluation Section

If you are doing evaluation you need to specify the type of environment you trained your agents in, the actual environment, the number of 
evaluation episodes, where you are loading the agent for evaluation from, what metrics you are going to measure by, and whether or not we're 
going to render the evaluation.

.. code-block:: ini

   [Evaluation]
   device               =   "cpu"
   expert_population    =   0.2
   num_evals            =   3
   num_envs             =   1
   eval_episodes        =   1


.. rubric:: Environment Section

Here you specify the type of environment, the name of the environment, whether or not we are rendering the environment, and normalization values.

.. code-block:: ini

   [Environment]
   device               =   'gpu'
   type                 =   'GymEnvironment'
   env_name             =   'CartPole-v0'
   episode_max_length   =   200
   expert_reward_range  =   {'CartPole-v0': [190, 200]}
   num_envs             =   1
   render               =   False
   port                 =   5010
   normalize            =   False
   reward_factor        =   0.1
   min_reward           =   0
   max_reward           =   1
   episodic_load_rate   =   1


.. rubric:: Admin Section

Here you need to specify where you want to save and if you want traceback warnings and if you do want to save then to what path.

File management settings. Where to save runs. Whether or not to save and if you want traceback warnings.

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


Agent Config: Agent-Cartpole.ini
================================

.. rubric:: Learner Section

Here you need to specify the type of learner, how many episodes to run, how often to save a checkpoint, whether or not we are loading agents 
in or training a fresh batch.

.. code-block:: ini

   [Learner]
   episodes                     = 5000
   evaluate                     = False
   load_agents                  = False
   save_checkpoint_episodes     = 50
   episodes_to_update           = 1
   n_traj_pulls                 = 5
   evolve                       = False
   initial_evolution_episodes   = 25
   evolution_episodes           = 125
   p_value                      = 0.05
   perturb_factor               = [0.8, 1.2]



.. rubric:: Algorithm Section

In the Algorithm section you need to specify the type of algorithm, whether or not you're using a replay buffer, the loss function, 
regularizer, whether or not we are using recurrence, epsilon greedy strategy, and hard update frequency.

.. code-block:: ini

   [Algorithm]
   type                 = "MADDPGAlgorithm"
   method               = "permutations"
   update_iterations    = 1
   loss_function        = 'MSELoss'
   gamma                = 0.999
   tau                  = 0.01



.. rubric:: Replay Buffer Section

You need to specify the type of the replay buffer you are using, the buffer's capacity or max size, and the size of the batch we'll be 
updating the networks on.

.. code-block:: ini

   [Buffer]
   type         = 'MultiTensorBuffer.MultiAgentTensorBuffer'
   capacity     = 10000
   batch_size   = 64


.. rubric:: Agent Section

For the agent we only specify the optimizer and the learning rate.

.. code-block:: ini

   [Agent]
   hp_random            = False
   lr_factors           = [1000, 10000]
   lr_uniform           = [1, 10]
   epsilon_range        = [0, 0.5]
   ou_range             = [0, 0.5]
   
   optimizer_function   = 'Adam'
   actor_learning_rate  = 0.001
   critic_learning_rate = 0.001
   lr_decay             = {'factor': 0.75, 'average_episodes': 50, 'wait_episodes_to_decay': 5}
   exploration_steps    = 1000
   
   actions_range        = [-1, 1]
   epsilon_start        = 0.95
   epsilon_end          = 0.01
   epsilon_episodes     = 500
   epsilon_decay_degree = 2
   
   noise_start          = 0.95
   noise_end            = 0.1
   noise_episodes       = 500
   noise_decay_degree   = 2


.. rubric:: Network Section

Here you need to specify the network structure.

.. code-block:: ini

   [Network]
   actor  = {'layers': [128], 'activation_function': ['ReLU'], 'output_function': None, 'last_layer': True}
   critic = {'layers': [128], 'activation_function': ['ReLU'], 'output_function': None, 'last_layer': True}


==========================
Running Unity Environments
==========================

We are using the `Python API UnityEnvironment <https://github.com/Unity-Technologies/ml-agents>`_ provided by 
Unity. To run on Shiva, all you need is the binary file for the scene you want to load for your environment. 
Here are some requirements with regards the build, and where/how to use the API.

For additional documentation, `Unity API <https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md>`_ 
is available if wanting to extend new features.

.. rubric:: Building Scenes on Unity Editor

* Recommended settings for the player under **Player Settings > :**
  - Run in background: True
  - Display Resolution Dialog: Disabled
* For the Prefabs/Agents
  - Make sure the **Behaviour Parameters** has no loaded model and it's empty
  - Stacked observations are supported
  - Only one brain is supported (for now)
  - Actions should come in one single branch, either continuous or discrete (no parametrized)
* The file for the **Scene_name.x86_64** extension should be placed in the **shiva/envs/unitybuilds/** and declared in the **exec** attribute for the config.

.. rubric:: Config Templates

The attributes set in the config will be accessible as a class attribute for the `UnityWrapperEnvironment class <https://github.com/nflux/Control-Tasks/blob/docs/shiva/shiva/envs/MultiAgentUnityWrapperEnv1.py>`_.

Here's the template

.. code-block:: ini

   [Environment]
   device = "gpu"
   type = 'MultiAgentUnityWrapperEnv1'
   exec = 'shiva/envs/unitybuilds/1/3DBall/3DBall.app'
   env_name = '3DBall'
   num_envs = 1
   episode_max_length = 1000
   episodic_load_rate = 1
   expert_reward_range = {'3DBall?team=0': [90, 100]}
   render = False
   port = 5010
   share_viewer = True
   normalize = False
   unity_configs = {}
   unity_props = {}

* Note that the **env_name** attribute must be the Brain name on Unity.

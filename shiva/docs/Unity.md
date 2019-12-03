# Using Shiva with Unity

We are using the [Python API UnityEnvironment](https://github.com/Unity-Technologies/ml-agents) provided by Unity. To run on Shiva, all you need is the binary file for the scene you want to load as your agent environment. Here are some requirements with regards the build, and where/how to use the API.

For additional documentation, [Unity API](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md) is available if wanting to extend new features.

**Building Scenes on Unity Editor**
* The settings that need to be selected before building are **Player Settings > :**
  - Run in background: True
  - Display Resolution Dialog: Disabled
* Prefabs/Agents
  - Make sure the _Behaviour Parameters_ has no loaded model and it's empty
  - Only one brain is supported and actions should come in one single branch (either continuous or discrete)
* The file for the **Scene_name.x86_64** extension should be placed in the **shiva/envs/unitybuilds/** and declared in the **exec** attribute for the config. See below the config.

**Config Templates**
  - The attribute setted in the config will be accessible as a class attribute for the [UnityWrapperEnvironment class](https://github.com/nflux/Control-Tasks/blob/master/shiva/shiva/envs/UnityWrapperEnvironment.py).
Here's the template for the config

```
[Environment]
type='UnityWrapperEnvironment'
exec='shiva/envs/unitybuilds/Scene_name/Scene_name.x86_64'
env_name='Scene_name'
train_mode = True
```

* Note that the **env_name** attribute must be the Brain name on Unity.

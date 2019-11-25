# Using Shiva with Unity

**Beta Feature**

At this moment we are currently using sockets to intercept experiences from the Unity Environment. We are working on a lower level interface with Unity. That being said we can set up an environment in Shiva that will allow us to run the algorithms we want in the environment. 

## Unity Editor

In the current version of this feature you need the Unity Editor open. The future version will use binary files. You can test your environments by simply creating your model inside of Shiva. You need to use a dummy brain in the agent and add a function in the agent's c sharp code that will communicate with Shiva. The best place to connect is somewhere where you have access to observations and where you can take action.  

## Available Unity Environments

* 3DBall 
* More to come soon

See below for instructions on how to connect Shiva to your Unity Environment.
## Connecting Shiva to your Unity Environment

Given that your config is set up appropriately ([how to set up a config file](url)) all you need to do to connect Shiva to Unity is to add our C-sharp client code in your agents file. We set up private variables store the observations when CollectObservations() is called and we added the socket client code in the ActorAction(). With that simple interception we are able to get a handle on the environment.

Here's how we did it in 3DBallAgent.cs for the 3DBall environment.
[Agent Code](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/envs/ml-agents/UnitySDK/Assets/ML-Agents/Examples/3DBall/Scripts/Ball3DAgent.cs)
 


# Configuration Files
___

We store all hyperparameters as well as data needed to run the models inside of .ini configuration files. In order to set up a model you need to configure each object needed for your pipeline. 

Shiva always requires Metalearner, Learners, Algorithms, Agents, Networks, Environments, perhaps Replay Buffers components to be able to run. As such the configuration file requires a section for each component.

Therefore your configuration file might look something like this:

; Test config for DQN
; NOTE: options within sections cannot have the same names
; some options are repeating, we need to get rid of duplicates

### MetaLearner Section
___
The Metalearner oversees the learners and will be able to do population based tuning on the hyperparameters so how you want the Metalearner to run would be configured here.

You specify the type of MetaLearner, the mode (production or evaluation), whether there is evolution, and if we are optimizing the hyperparameters.
```
[MetaLearner]
type='SingleAgentMetaLearner'
start_mode="production"
optimize_env_hp=False
optimize_learner_hp=False
evolution=False
```
You can see all the available MetaLearners [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/metalearners).

### Learner Section
___
Here you need to specify the type of learner, how many episodes to run, how often to save a checkpoint, whether or not we are loading agents in or training a fresh batch.
```
[Learner]
type='SingleAgentDQNLearner'
using_buffer=True
episodes=10_000
save_frequency=500
;metrics =  ["Reward", "LossPerStep", "TotalReward"]
;load_path='runs/ML-CartPole-v0-10-24-02:06/L-0'
load_path=False
```
You can see all the available Learners [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/learners).


### Algorithm Section
In the Algorithm section you need to specify the type of algorithm, whether or not you're using a replay buffer, the loss function, regularizer, whether or not we are using recurrence, epsilon greedy strategy, and hard update frequency.
___
```
[Algorithm]
type='DQNAlgorithm'
replay_buffer=True
loss_function='MSELoss'
regularizer=0
recurrence=False
gamma=0.99
beta=0
epsilon_start=1
epsilon_end=0.02
epsilon_decay=0.00005
c=200
```
You can see all the available Algorithms [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/algorithms).

## Environment Section
Here you specify the type of environment, the name of the environment, whether or not we are rendering the environment, and normalization values.
___
```
[Environment]
type='GymDiscreteEnvironment'
env_name='LunarLander-v2'
render=False
normalize=True
b=1
a=-1
min=-1
max=100
normalize=True
b=1
a=-1
min=-1
max=100
```
You can see all the available Environments [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/envs).

## Evaluation Section
If you are doing evaluation you need to specify the type of environment you trained your agents in, the actual environment, the number of evalueation episodes, where you are loading the agent for evaluation from, what metrics you are going to measure by, and whether or not we're going to render the evaluation.
___
```
[Evaluation]
env_type =          "Gym"
environment =       ["CartPole-v1"]
episodes =          10
load_path =         ["runs/ML-Gym-CartPole-v1-10-20-20:07/", "runs/ML-Gym-CartPole-v1-10-20-20:09/", "runs/ML-Gym-CartPole-v1-10-20-20:48/"]
metrics =           ["AveRewardPerEpisode", "MaxEpisodicReward", "MinEpisodicReward", "AveStepsPerEpisode"]
env_render =        True
```
You can see all the available Evaluation Environments [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/envs).

## Replay Buffer Section
___

You need to specify the type of the replay buffer you are using, the buffer's capacity or max size, and the size of the batch we'll be updating the networks on.

```
[Buffer]
type='SimpleBuffer'
capacity=100_000
batch_size=32
```
You can see all the available Replay Buffers [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/buffers).

## Agent Section
___
For the agent we only specify the optimizer and the learning rate.
```
[Agent]
optimizer_function='Adam'
learning_rate=0.003
```
You can see all the available Agents [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/agents).

## Network Section
___
Here you need to specify the network structure.
```
[Network]
network = {'layers': [400, 300], 'activation_function':["ReLU","ReLU"], 'output_function': None, 'last_layer': True}
```
You can see all the available Networks [here](https://github.com/nflux/Control-Tasks/tree/demo/shiva/shiva/networks).

## Admin Section
___

Here you need to specify where you want to save and if you want traceback warnings and if you do want to save then to what path.

File management settings. Where to save runs. Whether or not to save and if you want traceback warnings.
```
[Admin]
save =              True
traceback =         True
directory = {'runs': '/runs'}
```
You can see the source code [here](https://github.com/nflux/Control-Tasks/blob/demo/shiva/shiva/Shiva.py).

## Image Processing
___
Coming soon.
```
[ImageProcessing]
Not yet implemented.
```

## Video Processing
___
Coming soon.
```
[VideoProcessing]
Not yet implemented.
```

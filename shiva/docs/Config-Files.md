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
type = 'MPIPBTMetaLearner'
pbt = False
num_mevals = 1
learners_map = {'configs/MADDPG/Agent-3DBall.ini': ['3DBall?team=0']}
num_learners_maps = 1
num_menvs_per_learner_map = 1
```
You can see all the available MetaLearners [here](../shiva/metalearners).

### Learner Section
___
Here you need to specify the type of learner, how many episodes to run, how often to save a checkpoint, whether or not we are loading agents in or training a fresh batch.
```
[Learner]
episodes = 5000
evaluate = False
load_agents = False
save_checkpoint_episodes = 50
episodes_to_update = 1
n_traj_pulls = 5

evolve = False
initial_evolution_episodes = 25
evolution_episodes = 125
p_value = 0.05
perturb_factor = [0.8, 1.2]
```
You can see all the available Learners [here](../shiva/learners).


### Algorithm Section
In the Algorithm section you need to specify the type of algorithm, whether or not you're using a replay buffer, the loss function, regularizer, whether or not we are using recurrence, epsilon greedy strategy, and hard update frequency.
___
```
[Algorithm]
type = "MADDPGAlgorithm"
method = "permutations"
update_iterations = 1
loss_function = 'MSELoss'
gamma = 0.999
tau = 0.01
```
You can see all the available Algorithms [here](../shiva/algorithms).

## Environment Section
Here you specify the type of environment, the name of the environment, whether or not we are rendering the environment, and normalization values.
___
```
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
;reward_factor = 0.001
;min_reward = -1000
;max_reward = 2
unity_configs = {}
unity_props = {}
```
You can see all the available Environments [here](../shiva/envs).

## Evaluation Section
If you are doing evaluation you need to specify the type of environment you trained your agents in, the actual environment, the number of evalueation episodes, where you are loading the agent for evaluation from, what metrics you are going to measure by, and whether or not we're going to render the evaluation.
___
```
[Evaluation]
device = "cpu"
expert_population = 0.1
num_evals = 3
num_envs = 1
render = False
eval_episodes = 10
```
You can see all the available Evaluation Environments [here](../shiva/envs).

## Replay Buffer Section
___

You need to specify the type of the replay buffer you are using, the buffer's capacity or max size, and the size of the batch we'll be updating the networks on.

```
[Buffer]
type = 'MultiTensorBuffer.MultiAgentTensorBuffer'
capacity = 10000
batch_size = 64
```
You can see all the available Replay Buffers [here](../shiva/buffers).

## Agent Section
___
For the agent we only specify the optimizer and the learning rate.
```
[Agent]
hp_random = False
lr_factors = [1000, 10000]
lr_uniform = [1, 10]
epsilon_range = [0, 0.5]
ou_range = [0, 0.5]

optimizer_function = 'Adam'
actor_learning_rate = 0.001
critic_learning_rate = 0.001
lr_decay = {'factor': 0.75, 'average_episodes': 50, 'wait_episodes_to_decay': 5}
exploration_steps = 1000

actions_range = [-1, 1]
epsilon_start = 0.95
epsilon_end = 0.01
epsilon_episodes = 500
epsilon_decay_degree = 2

noise_start = 0.95
noise_end = 0.1
noise_episodes = 500
noise_decay_degree = 2
```
You can see all the available Agents [here](../shiva/agents).

## Network Section
___
Here you need to specify the network structure.
```
[Network]
actor = {'layers': [128], 'activation_function': ['ReLU'], 'output_function': None, 'last_layer': True}
critic = {'layers': [128], 'activation_function': ['ReLU'], 'output_function': None, 'last_layer': True}
```
You can see all the available Networks [here](../shiva/networks).

## Admin Section
___

Here you need to specify where you want to save and if you want traceback warnings and if you do want to save then to what path.

File management settings. Where to save runs. Whether or not to save and if you want traceback warnings.
```
[Admin]
iohandler_address   = 'localhost:50001'
print_debug         = True
save                = True
traceback           = True
directory           = {'runs': '/runs/Unity-3DBall/'}
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
    'Learner':      1,
    'Agent':        0,
    'Algorithm':    1,
    'MultiEnv':     1,
    'Env':          0,
    'EvalWrapper':  1,
    'Evaluation':   1,
    'EvalEnv':      0
    }

```
You can see the source code [here](../shiva/Shiva.py).

# Replay Buffers
Replay buffers that can be used in your algorithms.
## Config Requirements
Specified in detail below.

## Contents

* ReplayBuffer
[Go to Code](../shiva/buffers/ReplayBuffer.py)
* SimpleBuffer
[Go to Code](../shiva/buffers/SimpleBuffer.py)
* TensorBuffer
[Go to Code](../shiva/buffers/TensorBuffer.py)


## ReplayBuffer
___
[Go to Code](../shiva/buffers/ReplayBuffer.py)
### Config Setup
```
[Replay_Buffer]
type='ReplayBuffer'
capacity=100_000
batch_size = 128
num_agents=1
```

## SimpleBuffer
___
[Go to Code](../shiva/buffers/SimpleBuffer.py)
### Config Setup
```
[Replay_Buffer]
type='SimpleBuffer'
capacity=100_000
batch_size = 128
num_agents=1
```

## TensorBuffer
[Go to Code](../shiva/buffers/TensorBuffer.py)
___
### Config Setup
```
[Replay_Buffer]
type='TensorBuffer'
capacity=100_000
batch_size = 128
num_agents=1
```
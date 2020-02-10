from enum import IntEnum

class Tags(IntEnum):
    '''
        Tags to be used for Message Passing Interface
    '''
    configs = 0
    specs = 1

    # for Python List approach
    trajectory = 2

    # for Numpy approach
    trajectory_length = 2
    trajectory_observations = 3
    trajectory_actions = 4
    trajectory_rewards = 5
    trajectory_next_observations = 6
    trajectory_dones = 7

    new_agents = 10
    load_agents = 15
    evolution = 20
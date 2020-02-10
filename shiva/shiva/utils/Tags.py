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
    trajectory_info = 2
    trajectory_observations = 3
    trajectory_actions = 4
    trajectory_rewards = 5
    trajectory_next_observations = 6
    trajectory_dones = 7
    trajectory_eval = 8

    new_agents = 10
    #For updated rankings
    rankings = 11
    evolution = 20
    agent_id = 30
    evals = 40

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
    trajectory_actions_mask = 9
    trajectory_next_actions_mask = 10

    new_agents_request = 100
    new_agents = 101
    clear_buffers = 110
    #For updated rankings
    rankings = 120
    new_pbt_agents = 130
    save_agents = 140

    evolution = 200
    evolution_config = 210
    evolution_request = 220

    agent_id = 300
    evals = 400
    io_config = 500
    io_learner_request = 510
    io_menv_request = 520
    io_eval_request = 530

    close = 999
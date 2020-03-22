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
    new_agents_request=9
    new_agents = 10
    clear_buffers=11
    #For updated rankings
    rankings = 12
    new_pbt_agents = 13
    save_agents = 14
    evolution = 20
    evolution_config = 21
    agent_id = 30
    evals = 40
    io_config = 50
    io_learner_request = 51
    io_menv_request = 52
    io_eval_request = 53

    #io_checkpoint_save = 51
    #io_checkpoint_load =52
    #io_pbt_save = 53
    #io_pbt_load = 54
    #io_evo_load = 55
    #io_evals_save = 56
    #io_evals_load = 57
    #io_load_agents = 58
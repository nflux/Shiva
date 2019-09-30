from pretrain_env import pretrain_env
import os
import argparse
import numpy as np
# from run_pt import pretrain

def main(args):
    port = args.port
    left_side = 3
    right_side = 3
    fpt = 500
    untouched_time = 500
    sync_mode = True
    fullstate = True
    deterministic = True
    seed = np.random.randint(1000)
    #seed = 150
    start_viewer = True
    log_dir = args.log_dir

    # Control Random Initilization of Agents and Ball
    control_rand_init = True
    ball_x_min = -0.1
    ball_x_max = 0.1
    ball_y_min = -0.1
    ball_y_max = 0.1
    agents_x_min = -0.4
    agents_x_max = 0.4
    agents_y_min = -0.4
    agents_y_max = 0.4
    change_every_x = 1000000000
    change_agents_x = 0.01
    change_agents_y = 0.01
    change_balls_x = 0.01
    change_balls_y = 0.01

    offense_team_bin = args.offense_team
    defense_team_bin = args.defense_team

    if os.path.isdir(os.getcwd() + '/pretrain_data/log_' + str(port)):
        file_list = os.listdir(os.getcwd() + '/pretrain_data/log_' + str(port))
        [os.remove(os.getcwd() + '/pretrain_data/log_' + str(port) + '/' + f) for f in file_list]
    else:
        os.mkdir(os.getcwd() + '/pretrain_data/log_' + str(port))
    
    if os.path.isdir(os.getcwd() + '/pretrain_data/pt_logs_' + str(port)):
        file_list = os.listdir(os.getcwd() + '/pretrain_data/pt_logs_' + str(port))
        [os.remove(os.getcwd() + '/pretrain_data/pt_logs_' + str(port) + '/' + f) for f in file_list]
    else:
        os.mkdir(os.getcwd() + '/pretrain_data/pt_logs_' + str(port))

    pe = pretrain_env(num_TNPC=left_side, num_ONPC=right_side, fpt=fpt, untouched_time=untouched_time, port=port,
                        sync_mode=sync_mode, fullstate=fullstate, seed=seed, ball_x_min=ball_x_min,ball_x_max=ball_x_max,ball_y_min=ball_y_min,ball_y_max=ball_y_max,verbose = False,
                        hfo_log_game=False, rcss_log_game=False, log_dir=log_dir, agents_x_min=agents_x_min, agents_x_max=agents_x_max,
                        agents_y_min=agents_y_min, agents_y_max=agents_y_max, change_every_x=change_every_x,
                        change_agents_x=change_agents_x, change_agents_y=change_agents_y, change_balls_x=change_balls_x,
                        change_balls_y=change_balls_y, control_rand_init=control_rand_init, record=False, record_server=True,
                        offense_team_bin=offense_team_bin, defense_team_bin=defense_team_bin, deterministic=deterministic, start_viewer=start_viewer)

    while(True):
        pass # Pretraining

def parseArgs():
    p = argparse.ArgumentParser(description='Create pt files', 
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--port', dest='port', type=int, default=2000,
                    help='port number 2000 or above')
    p.add_argument('--log_dir', dest='log_dir', type=str, default='log_2000',
                    help='Location of rcg file')
    p.add_argument('--offense-team', dest='offense_team', type=str, default='base',
                    help='specifies what binary to run for offense')
    p.add_argument('--defense-team', dest='defense_team', type=str, default='base',
                    help='specifies what binary to run for defense')
    
    args = p.parse_args()

    return args

if __name__ == '__main__':
    main(parseArgs())

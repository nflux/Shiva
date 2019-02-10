from pretrain_env import pretrain_env
import os
# from run_pt import pretrain

port = 6000
left_side = 1
right_side = 1
fpt = 500
untouched_time = 500
sync_mode = True
fullstate = True
deterministic = True
seed = 123
start_viewer = False

# Control Random Initilization of Agents and Ball
control_rand_init = True
ball_x_min = -0.8
ball_x_max = 0.8
ball_y_min = -0.8
ball_y_max = 0.8
agents_x_min = -0.4
agents_x_max = 0.4
agents_y_min = -0.4
agents_y_max = 0.4
change_every_x = 1000000000
change_agents_x = 0.01
change_agents_y = 0.01
change_balls_x = 0.01
change_balls_y = 0.01

offense_team_bin = 'base'
defense_team_bin = 'base'

if os.path.isdir(os.getcwd() + '/pt_logs'):
    file_list = os.listdir(os.getcwd() + '/pt_logs')
    [os.remove(os.getcwd() + '/pt_logs/' + f) for f in file_list]
else:
    os.mkdir(os.getcwd() + '/pt_logs')

pe = pretrain_env(num_TNPC=left_side, num_ONPC=right_side, fpt=fpt, untouched_time=untouched_time, port=port,
                    sync_mode=sync_mode, fullstate=fullstate, seed=seed, ball_x_min=ball_x_min,ball_x_max=ball_x_max,ball_y_min=ball_y_min,ball_y_max=ball_y_max,verbose = False,
                    hfo_log_game=False, rcss_log_game=False, log_dir='log', agents_x_min=agents_x_min, agents_x_max=agents_x_max,
                    agents_y_min=agents_y_min, agents_y_max=agents_y_max, change_every_x=change_every_x,
                    change_agents_x=change_agents_x, change_agents_y=change_agents_y, change_balls_x=change_balls_x,
                    change_balls_y=change_balls_y, control_rand_init=control_rand_init, record=True,
                    offense_team_bin=offense_team_bin, defense_team_bin=defense_team_bin, deterministic=deterministic, start_viewer=start_viewer)

while(True):
    pass # Pretraining



import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/utils')
sys.path.append(os.getcwd() + '/modules')
sys.path.append(os.getcwd() + '/modules/robocup')

import time
import Shiva
import modules.Environment as Env
import robocup as rc

shiva = Shiva.ShivaAdmin(os.getcwd() + '/init.ini')
configs = shiva.get_inits()
env_conf = configs[0]['Environment']

rc_env = Env.initialize_env(env_conf)
rc_env.load_viewer()

time.sleep(3)
for ep_i in range(env_conf['num_ep']):

    for et_i in range(env_conf['ep_length']):
        
        left_obs = rc_env.get_observation()
        next_obs,rew,done = rc_env.step(rc_env.env.left_actions, rc_env.env.left_action_params)

        if done:
            break
        left_obs = next_obs
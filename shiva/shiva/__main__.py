# import sys
# import os
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'modules'))
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))

# from settings import shiva
# from MetaLearner import initialize_meta

# if __name__ == '__main__':
#     meta = initialize_meta(shiva.get_inits())

# # then you can have an overview of all the different learners running at same time
# # maybe we can make a gui or do something

import argparse
import os
import copy
import torch
import metalearners, learners, algorithms, envs, buffers
import helpers.misc as misc
import helpers.config_handler as ch

config_dir = os.getcwd() + '/configs/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help='Config file name')
    parser.add_argument("-n", "--name", required=True, type=str, help="Name of the run")
    args = parser.parse_args()

    main_dict = ch.load_config_file_2_dict(config_dir + args.config)
    temp_dict = copy.deepcopy(main_dict)
    types = temp_dict['Types']

    buffer_class = misc.handle_package(buffers, types['buffer'])
    buffer = buffer_class(temp_dict['Buffer'])
    env_class = misc.handle_package(envs, types['env'])
    env = env_class(temp_dict['Environment'])
    temp_dict['Algorithm']['observation_space'] = env.observation_space
    temp_dict['Algorithm']['action_space'] = env.action_space
    temp_dict['Algorithm']['loss_function'] = misc.handle_package(torch.nn, temp_dict['Algorithm']['loss_function'])
    temp_dict['Algorithm']['optimizer_function'] = misc.handle_package(torch.optim, temp_dict['Algorithm']['optimizer_function'])
    algo_class = misc.handle_package(algorithms, types['algorithm'])
    algo = algo_class(temp_dict['Algorithm'])
    temp_dict['Agent']['observation_space'] = algo.observation_space
    temp_dict['Agent']['action_space'] = algo.action_space
    temp_dict['Agent']['optimizer_function'] = algo.optimizer_function
    temp_dict['Agent']['learning_rate'] = algo.learning_rate
    agent = algo.create_agent(temp_dict['Agent'], temp_dict['Network'])
    metalearner_class = misc.handle_package(metalearners, types['metalearner'])
    meta = metalearner_class(algo, env, agent, [], buffer, temp_dict['MetaLearner'], temp_dict['Learner'], types['env'])
    # meta.create_learner(agent, env, algo, buffer, temp_dict['Learner'])
    meta.run()









# import sys
# import os
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'modules'))
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))

import argparse
import os
import copy
import torch
import metalearners
from Shiva import ShivaAdmin
import helpers.misc as misc
import helpers.config_handler as ch

config_dir = os.getcwd() + '/configs/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help='Config file name')
    parser.add_argument("-n", "--name", required=True, type=str, help="Name of the run")
    args = parser.parse_args()

    main_dict = ch.load_config_file_2_dict(config_dir + args.config)

    # global shiva
    # shiva = ShivaAdmin(main_dict['admin'])
    metalearner_class = misc.handle_package(metalearners, main_dict['MetaLearner']['type'])

    meta = metalearner_class(main_dict)




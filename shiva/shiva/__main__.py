import argparse
import os
import copy
import torch
from ShivaAdmin import ShivaAdmin
# import metalearners
from importlib import import_module
import helpers.misc as misc
import helpers.config_handler as ch

config_dir = os.getcwd() + '/configs/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help='Config file name')
    parser.add_argument("-n", "--name", required=False, type=str, help="Name of the run")
    args = parser.parse_args()

    main_dict = ch.load_config_file_2_dict(config_dir + args.config)

    shiva = ShivaAdmin(main_dict['Admin'])

    metalearner_module = import_module('metalearners')
    metalearner_class = misc.handle_package(metalearner_module, main_dict['MetaLearner']['type'])

    meta = metalearner_class(main_dict)
import sys, os, argparse
from shiva.core.admin import Admin
from shiva.helpers.config_handler import load_config_file_2_dict, load_class

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, type=str, help='Config file name')
parser.add_argument("-n", "--name", required=False, type=str, help="Name of the run")
args = parser.parse_args()

config_dir = os.getcwd() + '/configs/'
main_dict = load_config_file_2_dict(config_dir + args.config)

Admin.init(main_dict['Admin']) # Admin is instantiated at shiva.core.admin for project global access

metalearner_class = load_class("shiva.metalearners", main_dict['MetaLearner']['type'])
meta = metalearner_class(main_dict)
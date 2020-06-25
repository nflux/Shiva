import sys, os, argparse, traceback, datetime, time
from mpi4py import MPI
from shiva.core.admin import Admin, logger
from shiva.helpers.config_handler import load_config_file_2_dict, load_class
from shiva.helpers.misc import terminate_process

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", required=True, type=str, help='Config file name')
parser.add_argument("-n", "--name", required=False, type=str, help="Name of the run")
args = parser.parse_args()

def start_meta(metalearner_class, configs):
    try:
        meta = metalearner_class(configs)
    except Exception as e:
        msg = "<Meta> error: {}".format(traceback.format_exc())
        print(msg)
        logger.info(msg, True)
        terminate_process()
    finally:
        pass

configs = load_config_file_2_dict(os.path.join(os.getcwd(), 'configs', args.config))

if 'configs_set' in configs['MetaLearner']:
    _date, _time = str(datetime.datetime.now()).split()
    tmpst = _date[5:] + '-' + _time[0:5]
    _run_root_dir = '{}-{}-configs-{}'.format(tmpst, len(configs['MetaLearner']['configs_set'])).replace(':', '')
    for ix, c in enumerate(configs['MetaLearner']['configs_set']):
        print("\n%%% {} run %%%\n%%% {} %%%\n".format(ix+1, c))
        _run_type = c.split('/')[1] # like, 1U-5P, specifically for Profiling
        run_config = load_config_file_2_dict(os.path.join(os.getcwd(), 'configs', c))
        run_config['Admin']['directory']['runs'] = os.path.join('/', 'runs', _run_root_dir, 'run-{}-{}'.format(ix, _run_type))
        Admin.init(run_config['Admin'])  # Admin is instantiated at shiva.core.admin for project global access
        metalearner_class = load_class("shiva.metalearners", run_config['MetaLearner']['type'])
        start_meta(metalearner_class, run_config)
        print("\n%%% END WITH {} %%%".format(c))
        time.sleep(5)
else:
    Admin.init(configs['Admin'])  # Admin is instantiated at shiva.core.admin for project global access
    metalearner_class = load_class("shiva.metalearners", configs['MetaLearner']['type'])
    start_meta(metalearner_class, configs)

exit(0)
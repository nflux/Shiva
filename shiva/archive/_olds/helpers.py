import configparser, ast
import os, sys
import fnmatch
import traceback, warnings
import pickle, json
import numpy as np, torch
from datetime import datetime

def save_pickle_obj(obj, filename):
    '''
        Saves a Python object

        Input
            @obj        Instance to save
            @filename   Absolute path where to save including filename
    '''
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_obj(filename):
    '''
        Loads a Python object

        Input
            @filename   Absolute path to the file
    '''
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def save_to_json(data, filename):
    '''
        Saves data to JSON

        Input
            @data       Data to be serialized as a JSON
            @filename   Absolute path where to save including filename
    '''
    with open(filename, 'w') as handle:
        json.dump(data, handle)

def load_from_json(filename):
    '''
        Loads a JSON

        Input
            @filename   Absolute path to the file
    '''
    with open(filename, 'r') as handle:
        return json.load(handle)

def parse_configs(url: str) -> list:
    '''
        Returns a list of config files that are read at the given url
    '''
    _return = []
    for f in os.listdir(url):
        f = os.path.join(url, f)
        if os.path.isfile(f):
            _return.append(load_config_file_2_dict(f))
        else:
            for subf in os.listdir(os.path.join(url, f)):
                _return.append(load_config_file_2_dict(subf))
    return _return

def load_config_file_2_dict(_FILENAME: str) -> dict:
    '''
        Input
            directory where the .ini file is

        Converts a config file into a meaninful dictionary
            DataTypes that reads
            
                lists of the format [20,30,10], both integers and floats
                floats when a . is found
                booleans valid by configparser .getboolean()
                integer
                strings
                
    '''
    parser = configparser.ConfigParser()
    parser.read(_FILENAME)
    r = {}
    for _h in parser.sections():
        r[_h] = {}
        for _key in parser[_h]:
            r[_h][_key] = ast.literal_eval(parser[_h][_key])
    r['_filename_'] = _FILENAME
    return r

def is_iterable(ob):
    try:
        some_object_iterator = iter(ob)
        return True
    except TypeError as te:
        return False

def dtype_2_configstr(val: object):
    # string
    if type(val) == str:
        return "{}{}{}".format('"', val, '"')
    # iterable
    if is_iterable(val):
            return "{}".format(str(val))
    # boolean
    if type(val) == bool:
        return str(val)

    # if type(val) == float or type(val) == int
    return str(val)


def save_dict_2_config_file(config_dict: dict, file_path: str):
    config = configparser.ConfigParser()
    if type(config_dict) == list:
        assert False, "Not expecting a list"
    else:
        for section_name, attrs in config_dict.items():
            if is_iterable(attrs) and '_filename_' != section_name:
                config.add_section(section_name)
                for attr_name, attr_val in attrs.items():
                    config.set(section_name, attr_name, dtype_2_configstr(attr_val))
    with open(file_path, 'w') as configfile:
        config.write(configfile)

def make_dir(new_folder: str) -> str:
    # Implement another try block if there are Permission problems
    try:
        os.makedirs(new_folder)
    except FileExistsError:
        pass
    return new_folder

def make_dir_timestamp(new_folder: str) -> str:
    date, time = str(datetime.now()).split()
    new_folder = new_folder + date[5:] + '-' + time[0:5]
    return make_dir(new_folder)

def find_pattern_in_path(path, pattern):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def action2one_hot(action_space: int, action_idx: int) -> np.ndarray:
    '''
        Returns a one-hot encoded action numpy.ndarray
    '''
    z = np.zeros(action_space)
    z[action_idx] = 1
    return z

def action2one_hot_v(action_space: int, action_idx: int) -> torch.tensor:
    '''
        Returns a one-hot encoded action torch.tensor
    '''
    z = torch.zeros(action_space)
    z[action_idx] = 1
    return z

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    '''
        Utility for debugging
        Comment last line to enable/disable
    '''
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
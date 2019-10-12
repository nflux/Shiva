import configparser
import ast
from datetime import datetime
import os
import traceback, warnings, sys

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
    return r

def save_dict_2_config_file(config_dict: dict, file_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if type(config_dict) == list:
        assert False, "Not expecting a list"
    else:
        for section_name, attrs in config_dict.items():
            config.add_section(section_name)
            for attr_name, attr_val in attrs.items():
                config.set(section_name, attr_name, str(attr_val))

    # Writing our configuration file to 'example.cfg'
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

'''
    Utility for debugging
    Comment last line to enable/disable
'''

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
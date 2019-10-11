import configparser
import ast
from datetime import datetime
import os

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

def save_dict_2_config_file(config: dict, filename: str) -> configparser.ConfigParser:
    pass

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


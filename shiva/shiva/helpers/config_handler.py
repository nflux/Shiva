import configparser, ast
import os

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
    for _section in parser.sections():
        r[_section] = {}

        # if _section == 'Network':
        #     # need to grab the list of networks
        #     networks = []
        #     for _key in parser[_section]:
        #         networks = ast.literal_eval(parser[_section][_key])
        #         for network in networks:
        #             config = {}
        #             for __h in parser[network]:
        #                 config[__h] = ast.literal_eval(parser[network][__h])
        #             r[_section][network] = config

        # else:
        for _key in parser[_section]:
            r[_section][_key] = ast.literal_eval(parser[_section][_key])

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
import configparser, ast, os
from importlib import import_module

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

def load_class(module_path, file_name) -> object:
    if '.' in file_name:
        # if file contains multiple classes
        file, class_name = file_name.split('.')
        module_path = module_path + '.' + file
        module = import_module(module_path)
        cls = getattr(module, class_name)
    else:
        module_name = module_path + '.' + file_name
        module = import_module(module_name)
        cls = getattr(module, file_name)
    return cls

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
    assert len(list(parser.sections())) > 0, "Config {} is empty".format(_FILENAME)
    for _section in parser.sections():
        r[_section] = {}
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
# <<<<<<< HEAD
            if '_filename_' != section_name:
                if type(attrs) == dict:
                    config.add_section(section_name)
                    for attr_name, attr_val in attrs.items():
                        config.set(section_name, attr_name, dtype_2_configstr(attr_val))
                elif type(attrs) == list:
                    # usually some MultiEnv scattering list data comes in here
                    for at in attrs:
                        if type(at) == dict:
                            section_name = '-'.join([at['type'], str(at['id'])])
                            config.add_section(section_name)
                            for attr_name, attr_val in at.items():
                                config.set(section_name, attr_name, dtype_2_configstr(attr_val))
                        else:
                            print("Couldn't save {}, expecting dict not {}".format(at, type(at)))
                else:
                    print("Couldn't save section {} with attr {}".format(section_name, attrs))
# =======
#             if type(attrs) == list:
#                 attrs = attrs[0]
#             if is_iterable(attrs) and '_filename_' != section_name:
#                 config.add_section(section_name)
#                 for attr_name, attr_val in attrs.items():
#                     config.set(section_name, attr_name, dtype_2_configstr(attr_val))
# >>>>>>> robocup-pbt-mpi

    with open(file_path, 'w') as configfile:
        config.write(configfile)


def merge_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res

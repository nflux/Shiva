import configparser, ast, os
from importlib import import_module

from typing import List, Dict, Tuple, Any, Union

def parse_configs(url: str) -> List:
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

def load_class(module_path: str, file_name: str) -> object:
    """

    Args:
        module_path (str): module name
        file_name (str): file name containing the class we want to be imported.

    Returns:
        object: usually a class definition that wants to be instantiated
    """
    if '.' in file_name:
        # if file contains multiple classes
        file, class_name = file_name.split('.')
        module_path = module_path + '.' + file
        module = import_module(module_path)
        cls = getattr(module, class_name)
    else:
        # this works for importing examples such as
        #       from shiva.agents.DDPGAgent import DDPGAgent
        module_name = module_path + '.' + file_name
        module = import_module(module_name)
        cls = getattr(module, file_name)
    return cls

def load_config_file_2_dict(_FILENAME: str) -> Dict:
    """
    Converts the .ini file into a usable dictionary. All data types like int, float, bool, dict, lists, tuples seem to be supported.

    Args:
        _FILENAME: full/absolute path to the .ini file

    Returns:
        Dict
    """
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


def save_dict_2_config_file(config_dict: Dict, file_path: str) -> None:
    """
    Saves the given `config_dict` into a .ini file. Really good if used after using `load_config_file_2_dict`

    Args:
        config_dict (Dict): config dictionary to be saved
        file_path (str): file path where we want to save the .ini file

    Returns:
        None
    """
    config = configparser.ConfigParser()
    if type(config_dict) == list:
        assert False, "Not expecting a list"
    else:
        for section_name, attrs in config_dict.items():
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

    with open(file_path, 'w') as configfile:
        config.write(configfile)

def dtype_2_configstr(val: Union[str, Dict, List, bool, int, float]) -> str:
    """
    Converts the `val` into a string.

    Args:
        val (Union[str, Dict, List, bool, int, float]): object to be converted

    Returns:
        str: string value of `val`
    """
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

def is_iterable(ob: object) -> bool:
    """
    Check if the given object is an iterable or not.

    Args:
        ob (object): object to be evaluated

    Returns:
        bool
    """
    try:
        some_object_iterator = iter(ob)
        return True
    except TypeError as te:
        return False

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merge two dictionaries.

    Args:
        dict1 (Dict):
        dict2 (Dict):

    Returns:
        Dict
    """
    res = {**dict1, **dict2}
    return res

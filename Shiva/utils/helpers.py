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

import configparser

def config_file_to_dict(_FILENAME):
    _ITERABLE_LIMITS = ['[', ']']
    parser = configparser.ConfigParser()
    parser.read(_FILENAME)

    def try_iter(_val):
        if (_ITERABLE_LIMITS[0] in _val[0] and _ITERABLE_LIMITS[1] not in _val[-1]) or (_ITERABLE_LIMITS[0] not in _val[0] and _ITERABLE_LIMITS[1] in _val[-1]):
            assert False, 'Incorrect format of iterable in config file, got {}'.format(_val)
        if _ITERABLE_LIMITS[0] in _val[0] and _ITERABLE_LIMITS[1] in _val[-1]:
            _set = _val[1:-1]
            if '.' in _set:
                return list(map(float, _val[1:-1].split(',')))
            else:
                return list(map(int, _val[1:-1].split(',')))
        else:
            return False

    def get_dtype(_rval):
        val = try_iter(_rval)
        if type(val) == list: return val
        if '.' in _rval:
            try:
                return float(_rval)
            except:
                pass
        try:
            return _rval.getboolean()
        except:
            pass
        try:
            return int(_rval)
        except:
            pass
        try:
            return _rval[1:-1]
        except:
            assert False, 'datatype from dirs.ini could be read. got {}'.format(_rval)

    r = {}

    for _h in parser.sections():
        r[_h] = {}
        for _key in parser[_h]:
            r[_h][_key] = get_dtype(parser[_h][_key])
    return r
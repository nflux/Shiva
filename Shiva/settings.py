import configparser
import utils.helpers as helpers

_DIRS_INI = './Control-Tasks/Shiva/dirs.ini'

if __name__ == "__main__":
    dirs = helpers.config_to_dict(_DIRS_INI)
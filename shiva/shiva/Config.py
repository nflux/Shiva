import helpers.config_handler as ch

class Config:
    def __init__(self, dictionary):
        {setattr(self, k, v) for k,v in dictionary.items()}

def create_dict(conf_path):
    return ch.load_config_file_2_dict(conf_path)

def dict2Conf(mydict, key):
    return Config(mydict[key])
    

def add_attr(obj, config):
    pass
import ast
import torch
import datetime

def get_dict(config):
    conf_dict = {}

    sections = config.sections()

    for section in sections:
        options = config.options(section)
        for option in options:
            conf_dict[option] = ast.literal_eval(config.get(section, option))

    return conf_dict

class Config:
    def __init__(self, config):
        for k, v in config.items():
            setattr(self, k, v)


class config(Config):
    def __init__(self, config):
        self.conf_dict = get_dict(config)
        self.conditional_inits()
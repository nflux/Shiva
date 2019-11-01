import Config

def create_configs(confs):
    return {file:Config.Config(dictionary) for file,dictionary in confs}
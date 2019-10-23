import yaml
import os

class Config:
    def __init__(self, dictionary):
        {setattr(self, k, v) for k,v in dictionary.items()}
        self.dictionary = dictionary
    
    def save(self, filename):
        yaml.dump(self.dictionary, open(os.getcwd() +'/configs/yamls/'+ filename, 'w'), default_flow_style=False)
    

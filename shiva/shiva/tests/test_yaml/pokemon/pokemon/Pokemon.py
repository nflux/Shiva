class Pokemon(object):
    def __init__(self, config):
        for k,v in config.attacks.items():
            setattr(self, k,v)
        
        for k,v in config.stats.items():
            setattr(self, k,v)
    
    def speak(self):
        pass

class Pikachu(Pokemon):
    
    def speak(self):
        print('I am pika')

class Squirtle(Pokemon):

    def speak(self):
        print('I am squirt')
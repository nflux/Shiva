from pdb import set_trace as T
from .configs import Law, Chaos
import os

#Oversimplified user specification
#for 1-3 collaborators
# USER = 'your-username'
# if USER == 'your-username':
class Experiments:
   def __init__(self):
      #Thousandth
      self.prefix = 'test'
      self.remote = False
      self.local  = not remote

      self.test = True#local
      self.best = True#local
      self.load = True#local

      self.sample = not test
      self.singles = True
      self.tournaments = False
      
      self.exps = {}
      self.szs = [128]
      #For full distributed runs
      #szs = (16, 32, 64, 128)
      self.names = 'law chaos'.split()
      self.confs = (Law, Chaos)

   def makeExp(self,name, conf, sz, test=False):
      NENT, NPOP = sz, sz//16
      ROOT = 'resource/exps/' + name + '/'
      try:
         os.mkdir(ROOT)
         os.mkdir(ROOT + 'model')
         os.mkdir(ROOT + 'train')
         os.mkdir(ROOT + 'test')
      except FileExistsError:
         pass
      MODELDIR = ROOT + 'model/'

      exp = conf(self.remote, 
            NENT=NENT, NPOP=NPOP,
            MODELDIR=MODELDIR,
            SAMPLE=self.sample,
            BEST=self.best,
            LOAD=self.load,
            TEST=self.test)
      self.exps[name] = exp
      print(name, ', NENT: ', NENT, ', NPOP: ', NPOP)

   def makeExps(self):
      #Training runs
      for label, conf in zip(self.names, self.confs):
         for sz in szs:
            name = prefix + label + str(sz)
            makeExp(name, conf, sz, test=self.test)
          
   #Sample config
   # makeExps()
   # makeExp('sample', Chaos, 128, test=True)

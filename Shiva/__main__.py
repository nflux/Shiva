import sys
sys.path.append('./modules')
sys.path.append('./utils')

from settings import shiva
from modules.MetaLearner import initialize_meta

if __name__ == '__main__':
    meta = initialize_meta(shiva.get_inits())

# then you can have an overview of all the different learners running at same time
# maybe we can make a gui or do something
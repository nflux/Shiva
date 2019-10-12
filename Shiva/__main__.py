import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'modules'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils'))

from settings import shiva
from MetaLearner import initialize_meta

if __name__ == '__main__':
    meta = initialize_meta(shiva.get_inits())

# then you can have an overview of all the different learners running at same time
# maybe we can make a gui or do something
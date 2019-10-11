'''
    Utility for debugging
    Comment last line to enable/disable
'''
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

'''
    Shiva Testing Sandbox
'''

from MetaLearner import initialize_meta

# declare the path for the configs
ini_path = "Shiva/Initializers"

meta = initialize_meta(ini_path)


# then you can have an overview of all the different learners running at same time
# maybe we can make a gui or do something
# maybe we can set up tensorboard here with methods from meta



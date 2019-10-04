

'''

    Shiva Testing Sandbox

'''


# basically import shiva
from Meta_Learner import SingleAgentMetaLearner

# declare the path
ini_path = "Shiva/Initializers"


# then you pass the path to configs to the instance of the metalearner
meta = SingleAgentMetaLearner(ini_path)


# then you can have an overview of all the different learners running at same time
# maybe we can make a gui or do something
# maybe we can set up tensorboard here with methods from meta



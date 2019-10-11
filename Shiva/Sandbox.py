import settings
from modules.MetaLearner import initialize_meta

meta = initialize_meta(settings.ctr.init_path())

# then you can have an overview of all the different learners running at same time
# maybe we can make a gui or do something
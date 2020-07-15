from shiva.core.admin import Admin, logger

class MetaLearner(object):
    def __init__(self, configs, profile=True):
        {setattr(self, k, v) for k,v in configs['MetaLearner'].items()}
        self.configs = configs
        self.learnerCount = 0
        if profile:
            Admin.add_meta_profile(self, self.get_folder_name())

    # this would play with different hyperparameters until it found the optimal ones
    def exploit_explore(self):
        pass

    def genetic_crossover(self):
        pass

    def evolve(self):
        pass

    def evaluate(self):
        pass

    def record_metrics(self):
        pass

    def create_learner(self):
        raise NotImplemented

    def get_id(self):
        return self.get_new_learner_id()

    def get_new_learner_id(self):
        id = self.learnerCount
        self.learnerCount += 1
        return id

    def get_folder_name(self):
        try:
            folder_name = '-'.join([self.config['Algorithm']['type'], self.config['Environment']['env_name']])
        except:
            folder_name = '-'.join([self.config['Algorithm']['type1'], self.config['Environment']['env_name']])
        return folder_name

    def save(self):
        Admin.save(self)

    def log(self, msg, to_print=False, verbose_level=-1):
        '''If verbose_level is not given, by default will log'''
        if verbose_level <= self.configs['Admin']['log_verbosity']['MetaLearner']:
            text = "{}\t\t{}".format(str(self), msg)
            logger.info(text, to_print=to_print or self.configs['Admin']['print_debug'])

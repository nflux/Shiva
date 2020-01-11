from shiva.core.admin import Admin
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

class SingleAgentMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(SingleAgentMetaLearner, self).__init__(configs)
        self.configs = configs
        self.run()

    def run(self):

        if self.start_mode == self.EVAL_MODE:
            pass

        elif self.start_mode == self.PROD_MODE:

            self.learner = self.create_learner()

            Admin.add_learner_profile(self.learner)

            # try:
            self.learner.launch()
<<<<<<< HEAD
            
            # Admin.checkpoint(self.learner)
=======
            Admin.checkpoint(self.learner)
>>>>>>> dev

            self.learner.run()

            self.save()
            # except KeyboardInterrupt:
            #     print('Exiting for CTRL-C')
            # except Exception as inst:
            #     print(type(inst))  # the exception instance
            #     print(inst.args)  # arguments stored in .args
            # finally:
            #     print('Cleaning up possible extra learner processes')
            self.learner.close()

        print('bye')

    def create_learner(self):
        learner = load_class('shiva.learners', self.configs['Learner']['type'])
        if hasattr(self, 'start_port'):
            return learner(self.get_id(), self.configs, self.start_port)
        else:
            return learner(self.get_id(), self.configs)

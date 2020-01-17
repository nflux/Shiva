from shiva.core.admin import Admin, Communicator
from shiva.metalearners.MetaLearner import MetaLearner
from shiva.helpers.config_handler import load_class

class SingleAgentRPCMetaLearner(MetaLearner):
    def __init__(self, configs):
        super(SingleAgentRPCMetaLearner, self).__init__(configs)
        self.configs = configs
        self.run()

    def run(self):
        if self.start_mode == self.EVAL_MODE:
            pass

        elif self.start_mode == self.PROD_MODE:

            address = '[::]:50051'

            Communicator.start_env_server(address, self.configs)

            self.learner = self.create_learner()
            self.learner.env = Communicator.get_learner2env_client(self.learner.id, address, self.configs)
            Admin.add_learner_profile(self.learner)
            try:
                self.learner.launch()
                Admin.checkpoint(self.learner)
                self.learner.run()
                self.save()
            except KeyboardInterrupt:
                print('Exiting for CTRL-C')
            finally:
                self.close()
        print('bye')
        exit()

    def create_learner(self):
        learner = load_class('shiva.learners', self.configs['Learner']['type'])
        if hasattr(self, 'start_port'):
            return learner(self.get_id(), self.configs, self.start_port)
        else:
            return learner(self.get_id(), self.configs)


    def close(self):
        print('Cleaning up!')
        self.learner.close()
        Communicator.close_connections()
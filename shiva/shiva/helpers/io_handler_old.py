from mpi4py import MPI
import sys, time, traceback, os
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import numpy as np
from shiva.core.admin import logger
from shiva.helpers.misc import terminate_process
import shiva.helpers.file_handler as fh
from shiva.utils.Tags import Tags
from shiva.core.admin import Admin



class IOHandler(object):

    def __init__(self):
        self.meta = MPI.Comm.Get_parent()
        self.configs = self.meta.recv(None,source=0,tag=Tags.configs)
        self.learners_port = MPI.Open_port(MPI.INFO_NULL)
        self.menvs_port = MPI.Open_port(MPI.INFO_NULL)
        if self.configs['MetaLearner']['pbt']:
            self.evals_port = MPI.Open_port(MPI.INFO_NULL)
        self.specs = self._get_io_specs()
        self.log('Opened Ports: {}'.format(self.specs))
        self.meta.send(self.specs,dest=0,tag=Tags.io_config)
        self.log('Sent Meta my specs')
        self._connect_ports()
        self.log('Ports have been connected')
        self.info = MPI.Status()
        self.evo_evals = dict()
        self.run()

    def run(self):

        while True:
            time.sleep(0.001)
            self.service_learner_requests()
            self.service_menv_requests()
            if self.configs['MetaLearner']['pbt']:
                self.service_eval_requests()





    def _get_io_specs(self):
        if self.configs['MetaLearner']['pbt']:
            return {
            'learners_port': self.learners_port,
            'menvs_port': self.menvs_port,
            'evals_port': self.evals_port
            }
        else:
            return {
            'learners_port': self.learners_port,
            'menvs_port': self.menvs_port,
            }


    def _connect_ports(self):
        self.menvs = MPI.COMM_WORLD.Accept(self.menvs_port)
        self.log('MEnv Port Connected')
        self.learners = MPI.COMM_WORLD.Accept(self.learners_port)
        self.log('Learner Port Connected')
        if self.configs['MetaLearner']['pbt']:
            self.evals = MPI.COMM_WORLD.Accept(self.evals_port)
            self.log('Evals Port Connected')

    def service_learner_requests(self):
        if self.learners.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_checkpoint_save):
            #self.log('Detected Learner Checkpoint Request')
            self.save_learner_agents()
            #self.log('Learner Checkpoint Request Complete')

        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.io_pbt_save):
            #self.log('Detected PBT Save request')
            self.save_pbt_agents()
            #self.log('PBT save request complete')

        if self.learners.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.io_evo_load):
            #self.log('Detected evolution agent request')
            self.load_evolution_agent()
            #self.log('Evolution agent request complete')

        if self.learners.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_evals_load):
            #self.log('Detected evaluation request')
            self.load_evaluations()
            #self.log('Evaluation request complete')

    def service_menv_requests(self):
        if self.menvs.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_checkpoint_load):
            #self.log('Detected Checkpoint request')
            self.load_agent_checkpoint()
            #self.log('Checkpoint request complete')

        if self.menvs.Iprobe(source=MPI.ANY_SOURCE, tag=Tags.io_load_agents):
            #self.log('Detected  MultiEnv agent load request')
            self.load_agents()
            #self.log('Agent MultiEnv load request complete')

    def service_eval_requests(self):
        if self.evals.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_load_agents):
            #self.log('Detected Eval agent load request')
            self.load_eval_agents()
            #self.log('Agent Eval load request complete')

        if self.evals.Iprobe(source=MPI.ANY_SOURCE,tag=Tags.io_evals_save):
            #self.log('Detected Evaluation save request')
            self.save_evals()
            #self.log('Evaluation save request complete')


    def save_learner_agents(self):
        learner_dict = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.io_checkpoint_save)
        agents = [None] * len(learner_dict['agents'])

        #Admin.checkpoint(learner_dict['learner'], learner_dict['checkpoint_num'], learner_dict['function_only'], learner_dict['use_temp_folder'])
        for i in range(len(learner_dict['agents'])):
            agent_path = os.path.join(learner_dict['checkpoint_path'],learner_dict['agent_dir'][i].format(id=str(learner_dict['agents'][i].id)))
            fh.save_pickle_obj(learner_dict['agents'][i], os.path.join(agent_path, 'agent_cls.pickle'))
            learner_dict['agents'][i].save(agent_path, learner_dict['checkpoint_num'])


    def save_pbt_agents(self):
        learner_dict = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.io_pbt_save)
        for agent in learner_dict['agents']:
            agent_path = learner_dict['path'] + str(agent.id)
            agent.save(agent_path,0)
            fh.save_pickle_obj(agent, os.path.join(agent_path, 'agent_cls.pickle'))

    def load_evolution_agent(self):
        path = self.learners.recv(None, source=MPI.ANY_SOURCE, tag=Tags.io_evo_load,status=self.info)
        source = self.info.Get_source()
        agent = Admin._load_agents(path)[0]
        self.learners.send(agent,dest=source,tag=Tags.io_evo_load)

    def load_evaluations(self):
        eval_config = self.learners.recv(None,source=MPI.ANY_SOURCE, tag=Tags.io_evals_load,status=self.info)
        source = self.info.Get_source()
        self.log('Eval Config: {}'.format(eval_config))
        self.evo_evals['evals'] = np.load(eval_config['evals_path'])
        self.evo_evals['evo_evals'] = np.load(eval_config['evo_evals_path'])
        self.evo_evals['evo_agent'] = Admin._load_agents(eval_config['evo_agent_path'])[0]
        self.learners.send(self.evo_evals,dest=source,tag=Tags.io_evals_load)

    def load_agent_checkpoint(self):
        learner_spec = self.menvs.recv(None, source=MPI.ANY_SOURCE, tag=Tags.io_checkpoint_load,status=self.info)
        source = self.info.Get_source()
        agent = Admin._load_agents(learner_spec['load_path'])[0]
        self.menvs.send(agent,dest=source,tag=Tags.io_checkpoint_load)

    def load_agents(self):
        learners_specs = self.menvs.recv(None, source = MPI.ANY_SOURCE, tag=Tags.io_load_agents,status=self.info)
        source = self.info.Get_source()
        agents = [ Admin._load_agents(learner_specs['load_path'])[0] for learner_specs in learners_specs ]
        self.menvs.send(agents,dest=source,tag=Tags.io_load_agents)

    def load_eval_agents(self):
        agent_paths = self.evals.recv(None,source=MPI.ANY_SOURCE, tag=Tags.io_load_agents,status=self.info)
        self.log('Step 1 complete: {}'.format(agent_paths))
        source = self.info.Get_source()
        self.log('Step 2 complete: {}'.format(source))
        agents = [Admin._load_agents(path)[0] for path in agent_paths]
        self.log('Step 3 complete: {}'.format(agents))
        self.evals.send(agents,dest=source,tag=Tags.io_load_agents)
        self.log('Done')

    def save_evals(self):
        ep_evals = self.evals.recv(None, source=MPI.ANY_SOURCE,tag=Tags.io_evals_save,status=self.info)
        source = self.info.Get_source()
        agents = [None] * len(ep_evals['agent_ids'])
        for i in range(len(ep_evals['agent_ids'])):
            save_path = ep_evals['path']+'Agent_'+str(ep_evals['agent_ids'][i])
            load_path = ep_evals['path']+'Agent_'+str(ep_evals['new_agent_ids'][i])
            agents[i] = Admin._load_agents(load_path)[0]
            np.save(save_path+'/episode_evaluations',ep_evals['evals'][i])
        self.evals.send(agents,dest=source,tag=Tags.io_evals_save)

    def log(self, msg, to_print=False):
        text = 'IOHandler: {}'.format(msg)
        logger.info(text, True)










if __name__ == "__main__":
    try:
        IOHandler()
    except Exception as e:
        print("Eval Wrapper error:", traceback.format_exc())
    finally:
        terminate_process()

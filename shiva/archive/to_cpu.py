# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-p", "--path", required=True, type=str, help='Config file name')

# args = parser.parse_args()
import sys
from pathlib import Path
# __file__ = '/ez/src/Control-Tasks/shiva'
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
import torch
import pickle
from file_handler import save_pickle_obj, load_pickle_obj
pickles = ['agent_cls.pickle']
nets = ['actor', 'target_actor']
runs = '../../runs'
urls = ['/particle/tag/MADDPGAlgorithm-simple_tag.py-03-10-07:56/L0/Ep15000/0-agent 3/',
        '/particle/tag/MADDPGAlgorithm-simple_tag.py-03-10-07:56/L1/Ep15000/0-agent 0/',
        '/particle/tag/MADDPGAlgorithm-simple_tag.py-03-10-07:56/L1/Ep15000/0-agent 1/',
        '/particle/tag/MADDPGAlgorithm-simple_tag.py-03-10-07:56/L1/Ep15000/0-agent 2/'
       ]
for url in urls:
    agent_file_name = runs+url+'agent_cls.pickle'
    print("On load {}".format(agent_file_name))
    cls = load_pickle_obj(agent_file_name)
    print("Loaded")
    # change all nets to cpu
    for net_name in nets:
        path = runs+url
        nn = torch.load(path+net_name+'.pth').to('cpu')
        print("\tNet loaded {}".format(net_name))
        torch.save(nn, path+'cpu_'+net_name)
        print("\tNet saved {}".format(net_name))
        cls_net = getattr(cls, net_name).to('cpu')
        setattr(cls, net_name, cls_net)
        if hasattr(cls, 'critic'):
            delattr(cls, 'critic')
        if hasattr(cls, 'target_critic'):
            delattr(cls, 'target_critic')
        if hasattr(cls, 'actor_optimizer'):
            delattr(cls, 'actor_optimizer')

    save_file_name = runs+url+'cpu_agent_cls.pickle'
    print("To save {}".format(cls))
    save_pickle_obj(cls, save_file_name)
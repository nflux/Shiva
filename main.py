import os, argparse, configparser, config
import algorithms.maddpg as mad_algo
import pretrain.pretrain_process as pretrainer
import envs.rc_soccer.multi_envs as menvs
import torch.multiprocessing as mp
import algorithms.updates as updates

def parseArgs():
    parser =  argparse.ArgumentParser('Team Shiva')
    parser.add_argument('--env', type=str, default='rc',
                        help='type of environment')
    parser.add_argument('--conf', type=str, default='rc_test.ini')

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    config_parse = configparser.ConfigParser()

    conf_path = os.getcwd() + '/configs/' + args.conf

    # if args.env == 'rc':
    mp.set_start_method('forkserver',force=True)
    config_parse.read(conf_path)
    config = config.RoboConfig(config_parse)

    env = menvs.RoboEnvs(config)
    maddpg = mad_algo.init(config, env.template_env)
    update = updates.Update(config, env.team_replay_buffer, env.opp_replay_buffer)

    #Pretraining **Needs create_pretrain_files.py to test, importing HFO issue
    pretrainer = pretrainer.pretrain(config, env)

    # -------------Done pretraining actor/critic ---------------------------------------------
    maddpg.save_agent2d(config.load_path,0,config.load_same_agent,maddpg.torch_device)
    [maddpg.save_ensemble(config.ensemble_path,0,i,config.load_same_agent,maddpg.torch_device) for i in range(config.num_left)] # Save agent2d into ensembles

    maddpg.scale_beta(config.init_beta) 
    env.run()
    update.main_update(env, maddpg)
    

    


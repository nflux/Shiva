import os, argparse, configparser
import envs.rc_soccer.rc_envs as rc_envs

def parseArgs():
    parser =  argparse.ArgumentParser('Team Shiva')
    parser.add_argument('--env', type=str, default='rc',
                        help='type of environment')
    parser.add_argument('--conf', type=str, default='rc_cpu')

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    config_parse = configparser.ConfigParser()
    conf_path = os.getcwd() + '/configs/'

    if args.env == 'rc':
        rc_path = conf_path + 'rc_soccer/' + args.conf + '.ini'
        if not os.path.isfile(rc_path):
            print('Incorrect configuration provided')
            exit(0)
        config_parse.read(rc_path)
        envs = rc_envs.RoboEnvsWrapper(config_parse)
        envs.run()
    elif args.env == 'nmmo':
        # TO-DO
        print('Running nmmo')
    else:
        print('Invalid environment')


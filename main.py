import os, argparse, configparser
import envs.rc_soccer.rc_envs as rc_envs

def parseArgs():
    parser =  argparse.ArgumentParser('Team Shiva')
    parser.add_argument('--env', type=str, default='rc',
                        help='type of environment')
    parser.add_argument('--conf', type=str, default='')

    return parser.parse_args()

if __name__ == "__main__":
    args = parseArgs()
    config_parse = configparser.ConfigParser()
    conf_path = os.getcwd() + '/configs/' + args.conf
    if not os.path.isfile(conf_path):
        print('Incorrect configuration provided')
        exit(0)
    config_parse.read(conf_path)

    if args.env == 'rc':
        envs = rc_envs.RoboEnvsWrapper(config_parse)
        envs.run()
    else if args.env == 'nmmo':
        # TO-DO
        print('Running nmmo')
    else:
        print('Invalid environment')


import configparser, ast
import os, subprocess, time, signal
from HFO import hfo
import socket
from HFO.bin import Communicator

def load_config_file_2_dict(_FILENAME: str) -> dict:
    '''
        Input
            directory where the .ini file is

        Converts a config file into a meaninful dictionary
            DataTypes that reads
            
                lists of the format [20,30,10], both integers and floats
                floats when a . is found
                booleans valid by configparser .getboolean()
                integer
                strings
                
    '''
    parser = configparser.ConfigParser()
    parser.read(_FILENAME)
    r = {}
    for _section in parser.sections():
        r[_section] = {}
        for _key in parser[_section]:
            r[_section][_key] = ast.literal_eval(parser[_section][_key])
    r['_filename_'] = _FILENAME
    return r

class Bots:
    def __init__(self, config):
        {setattr(self, k, v) for k,v in config.items()}
        self.viewer = None
        self._comm = Communicator.ClientCommunicator(port=6001)
        self._comm.sendMsg('(init (version 8.0))')
        self._comm.sendMsg('(ear on)')
        # self._comm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self._comm.settimeout(None)
        # msg, addr = self._comm.recvfrom(8192)
        # print(msg, addr)
        # self._comm = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self._comm.bind(('localhost', 6001))
        # self._comm.listen()
        # self.conn, addr = self._comm.accept()
        # print('Accepted', addr)

    
    # found from https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    def _start_hfo_server(self):

            cmd = hfo.get_hfo_path() + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                  " --log-dir %s --message-size 256 --tackle-cycles 1 --no-offside --run-imit --offside-area-size 0"\
                  % (self.fpt, self.untouched_time, self.leftagents,
                     self.rightagents, self.leftbots, self.rightbots, self.port,
                     self.offense_on_ball, self.seed, self.ball_x_min, self.ball_x_max,
                     self.ball_y_min, self.ball_y_max, self.log_dir)
            
            if self.leftbots > 0:   cmd += " --offense-team %s" \
                % (self.offense_team_bin)
            if self.rightbots > 0:   cmd += " --defense-team %s" \
                % (self.defense_team_bin)
            if not self.sync_mode:      cmd += " --no-sync"
            if self.fullstate:          cmd += " --fullstate"
            if self.deterministic:      cmd += " --deterministic"
            if self.verbose:            cmd += " --verbose"
            if not self.rcss_log_game:  cmd += " --no-logging"
            if self.hfo_log_game:       cmd += " --hfo-logging"
            if self.record:             cmd += " --record"
            if self.record_server:      cmd += " --log-gen-pt"
            if self.control_rand_init:
                cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                        " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                        " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                        % (self.agents_x_min, self.agents_x_max, self.agents_y_min, self.agents_y_max,
                            self.change_every_x, self.change_agents_x, self.change_agents_y,
                            self.change_balls_x, self.change_balls_y)

            print('Starting server with command: %s' % cmd)
            self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
            time.sleep(1) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)
        cmd = hfo.get_viewer_path() +\
              " --connect --port %d" % (self.port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)
    
    def run(self):
        self._start_hfo_server()
        if self.start_viewer:
            self._start_viewer()
        
        while True:
            # self._comm.sendMsg('(think)')
            pass


if __name__ == "__main__":
    config_file = os.getcwd() + '/bot.ini'
    config = load_config_file_2_dict(config_file)
    bots = Bots(config['BotEnv'])
    bots.run()



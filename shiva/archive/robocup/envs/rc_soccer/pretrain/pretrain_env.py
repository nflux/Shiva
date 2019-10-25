from HFO import get_config_path, get_hfo_path, get_viewer_path
import os, subprocess, time, signal


class pretrain_env():
    """rc_env() extends the HFO environment to allow for centralized execution.

    Attributes:
        num_TA (int): Number of teammate agents. (0-11)
        num_OA (int): Number of opponent agents. (0-11)
        num_NPC (int): Number of opponent NPCs. (0-11)
        team_actions (list): List contains the current timesteps action for each
            agent. Takes value between 0 - num_states and is converted to HFO action by
            action_list.
        action_list (list): Contains the mapping from numer action value to HFO action.
        team_should_act (list of bools): Contains a boolean flag for each agent. Is
            activated to be True by Step function and becomes false when agent acts.
        team_should_act_flag (bool): Boolean flag is True if agents have
            unperformed actions, becomes False if all agents have acted.
        team_obs (list): List containing obs for each agent.
        team_obs_previous (list): List containing obs for each agent at previous timestep
        team_rewards (list): List containing reward for each agent
        start (bool): Once all agents have been launched, allows threads to listen for
            actions to take.
        world_states (list): Contains the status of the HFO world.
        team_envs (list of HFOEnvironment objects): Contains an HFOEnvironment object
            for each agent on team.
        opp_xxx attributes: Extend the same functionality for user controlled team
            to opposing team.

    Todo:
        * Functionality for synchronizing team actions with opponent team actions
        """
    # class constructor
    def __init__(self, num_TNPC = 1, num_ONPC = 1, fpt = 100, untouched_time = 100, sync_mode = True, port = 6000,
                 offense_on_ball=0, fullstate = False, seed = 123, pt_log_dir="/pretrain_data",
                 ball_x_min = -0.8, ball_x_max = 0.8, ball_y_min = -0.8, ball_y_max = 0.8,
                 verbose = False, rcss_log_game=False, hfo_log_game=False, log_dir="log",
                 agents_x_min=-0.8, agents_x_max=0.8, agents_y_min=-0.8, agents_y_max=0.8,
                 change_every_x=5, change_agents_x=0.1, change_agents_y=0.1, change_balls_x=0.1,
                 change_balls_y=0.1, control_rand_init=False,record=False, record_server=True,
                 defense_team_bin='base', offense_team_bin='base', deterministic=True, start_viewer=False):
        

        """ Initializes HFO_Env

        Args:
            num_TA (int): Number of teammate agents. (0-11)
            num_OA (int): Number of opponent agents. (0-11)
            num_ONPC (int): Number of opponent NPCs. (0-11)
            base (str): Which side for the team. ('base_left','base_right') NOTE: Keep this at base_left
            goalie (bool): Should team use a goalie. (True,False)
            num_trials (int): Number of episodes
            fpt (int): Frames per trial
            feat_lvl (str): High or low feature level. ('high','low')
            act_lvl (str): High or low action level. ('high','low')

        Returns:
            HFO_Env

        """
        self.log_dir = log_dir
        self.port = port
        self.hfo_path = get_hfo_path()
        self._start_hfo_server(frames_per_trial = fpt, untouched_time = untouched_time,
                                    offense_agents = 0, defense_agents = 0,
                                    offense_npcs = num_TNPC, defense_npcs = num_ONPC,
                                    sync_mode = sync_mode, port = port,
                                    offense_on_ball = offense_on_ball,
                                    fullstate = fullstate, seed = seed,
                                    ball_x_min = ball_x_min, ball_x_max = ball_x_max,
                                    ball_y_min= ball_y_min, ball_y_max= ball_y_max,
                                    verbose = verbose, rcss_log_game = rcss_log_game, 
                                    hfo_log_game=hfo_log_game, log_dir = log_dir, pt_log_dir=pt_log_dir,
                                    agents_x_min=agents_x_min, agents_x_max=agents_x_max,
                                    agents_y_min=agents_y_min, agents_y_max=agents_y_max,
                                    change_every_x=change_every_x, change_agents_x=change_agents_x,
                                    change_agents_y=change_agents_y, change_balls_x=change_balls_x,
                                    change_balls_y=change_balls_y, control_rand_init=control_rand_init,record=record, record_server=record_server,
                                    defense_team_bin=defense_team_bin, offense_team_bin=offense_team_bin, deterministic=deterministic)

        self.viewer = None

        if start_viewer:
            self._start_viewer() 
            
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    # found from https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
    def _start_hfo_server(self, frames_per_trial=100,
                              untouched_time=100, defense_agents=0,
                              offense_agents=0, offense_npcs=0,
                              defense_npcs=0, sync_mode=True, port=6000,
                              offense_on_ball=0, fullstate=False, seed=123,
                              ball_x_min=-0.8, ball_x_max=0.8,
                              ball_y_min=-0.8, ball_y_max=0.8,
                              verbose=False, rcss_log_game=False,
                              log_dir="log", pt_log_dir="/pretrain_data",
                              hfo_log_game=True,
                              agents_x_min=0.0, agents_x_max=0.0,
                              agents_y_min=0.0, agents_y_max=0.0,
                              change_every_x=1, change_agents_x=0.1,
                              change_agents_y=0.1, change_balls_x=0.1,
                              change_balls_y=0.1, control_rand_init=False,record=False, record_server=True,
                              defense_team_bin='base', offense_team_bin='helios16', deterministic=True):
            """
            Starts the Half-Field-Offense server.
            frames_per_trial: Episodes end after this many steps.
            untouched_time: Episodes end if the ball is untouched for this many steps.
            offense_agents: Number of user-controlled offensive players.
            defense_agents: Number of user-controlled defenders.
            offense_npcs: Number of offensive bots.
            defense_npcs: Number of defense bots.
            sync_mode: Disabling sync mode runs server in real time (SLOW!).
            port: Port to start the server on.
            offense_on_ball: Player to give the ball to at beginning of episode.
            fullstate: Enable noise-free perception.
            seed: Seed the starting positions of the players and ball.
            ball_x_[min/max]: Initialize the ball this far downfield: [-1,1]
            verbose: Verbose server messages.
            log_game: Enable game logging. Logs can be used for replay + visualization.
            log_dir: Directory to place game logs (*.rcg).
            """
            self.server_port = port
            cmd = self.hfo_path + \
                  " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
                  " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
                  " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
                  " --ball-x-max %f --ball-y-min %f --ball-y-max %f"\
                  " --log-dir %s --log-dir-pt %s --message-size 256 --tackle-cycles 1 --no-offside --offside-area-size 0"\
                  % (frames_per_trial, untouched_time, offense_agents,
                     defense_agents, offense_npcs, defense_npcs, port,
                     offense_on_ball, seed, ball_x_min, ball_x_max,
                     ball_y_min, ball_y_max, log_dir, pt_log_dir)
            #Adds the binaries when offense and defense npcs are in play, must be changed to add agent vs binary npc
            if offense_npcs > 0:   cmd += " --offense-team %s" \
                % (offense_team_bin)
            if defense_npcs > 0:   cmd += " --defense-team %s" \
                % (defense_team_bin)
            if not sync_mode:      cmd += " --no-sync"
            if fullstate:          cmd += " --fullstate"
            if deterministic:      cmd += " --deterministic"
            if verbose:            cmd += " --verbose"
            if not rcss_log_game:  cmd += " --no-logging"
            if hfo_log_game:       cmd += " --hfo-logging"
            if record:             cmd += " --record"
            if record_server:      cmd += " --log-gen-pt" 
            if control_rand_init:
                cmd += " --agents-x-min %f --agents-x-max %f --agents-y-min %f --agents-y-max %f"\
                        " --change-every-x-ep %i --change-agents-x %f --change-agents-y %f"\
                        " --change-balls-x %f --change-balls-y %f --control-rand-init"\
                        % (agents_x_min, agents_x_max, agents_y_min, agents_y_max,
                            change_every_x, change_agents_x, change_agents_y,
                            change_balls_x, change_balls_y)

            print('Starting server with command: %s' % cmd)
            self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
            time.sleep(3) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)
        cmd = get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)
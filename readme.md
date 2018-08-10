The test bench notebook 'DRQN_1_vs_0' runs a simple DQN with RNN for one offense-agent using high level actions [dribble,go_to_ball, reorient & shoot]. 
make sure to modify the config directory based on the HFO output when you run this notebook.
before running the notebook, make sure the HFO rcssserver is up, for instance: 
./bin/HFO --offense-agents=1 --seed 123 --headless --trials 10000 --frames-per-trial 100

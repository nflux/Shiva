#!/usr/bin/env python
# This Script runs our benchmark scripts


# GYM Based Testing
python ./shiva -c DQN.ini -n N

python ./shiva -c PPO.ini -n N

python ./shiva -c DDPG-Continuous.ini -n N


# UNITY Based Testing

python ./shiva -c DDPG-3DBall.ini -n N

python ./shiva -c DQN-Unity-Basic.ini -n N



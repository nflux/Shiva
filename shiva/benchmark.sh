#!/usr/bin/env python
# This Script runs our benchmark scripts

echo 'Benchmark Starting'
read -p 'About to remove OLD CSV Files are you sure you want to do this? y = YES n = No' yn
    case $yn in
        [Yy]* ) cd benchmarkCode; rm -rfv /Agent; tm -rfv /Algorithm; rm -rfv /Rewards; break;;
        [Nn]* ) break;; 
        * ) echo "Please answer y = YES or N = NO.";;
    esac


# if[[$REPLY =~ ^[Yy]$]]
# then
#     [[cd benchmarkCode
#     rm -rfv /Agent
#     rm -rfv /Algorithm
#     rm -rfv /Rewards]]
# fi
# GYM Based Testing
echo '--------------------------Running DQN.ini-------------------------'
python ./shiva -c DQN.ini -n N
echo '-------------------------Finished DQN.ini-------------------------'

echo '-------------------------Running PPO.ini-------------------------'
# python ./shiva -c PPO.ini -n N
echo '-------------------------Finished PPO.ini-------------------------'

echo '-------------------------Running DDPG-Continous.ini-------------------------'
# python ./shiva -c DDPG-Continuous.ini -n N
echo '-------------------------Finished DDPG-Continous.ini-------------------------'

# UNITY Based Testing
echo '-------------------------Running DDPG-3DBall.ini-------------------------'
# python ./shiva -c DDPG-3DBall.ini -n N
echo '-------------------------Finished DDPG-3DBall.ini-------------------------'

# python ./shiva -c DQN-Unity-Basic.ini -n N
echo '-------------------------Finished DDPG-3DBall.ini-------------------------'

# cd shiva/benchmarkCode
#  python benchAnalysis.py
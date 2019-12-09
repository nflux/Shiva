#!/usr/bin/env python

# This script is meant to install all the components of Shiva
cd ../shiva/shiva/envs/robocup/

# clone the robocup files
git clone https://github.com/nflux/rcssserver.git
git clone https://github.com/nflux/librcsc.git

# Install cmake, C++ Boost Libraries, Flex and Qt version 4
sudo apt-get install cmake libboost-dev libboost-all-dev flex qt4-default
cd HFO

#The above command will create a build/ folder with various executable files
bash compile.sh
​
## Ready to launch a RoboCup session

## Run pip off req.txt
pip install -r requirements.txt

### Install cuda and video drivers here
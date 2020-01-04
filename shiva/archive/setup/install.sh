#! /usr/bin/env bash

# This script is meant to install all the components of Shiva
​
## Setup on Ubuntu 16.04

# ​Go to robocup folder
cd ../shiva/shiva/envs/robocup/

install git
sudo apt-get install git

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

# cd shiva/shiva/envs/ml-agents/ml-agents-envs/

# Install Unity Python API
cd ../../ml-agents/ml-agents-envs/
pip install -e ./
cd ..
cd ml-agents
pip install -e ./

## Unity API installed


### Install cuda and video drivers here


# Python dependencies
pip instal gym tensorboard tensorboardX tensorflow pandas torch pynput jupyter
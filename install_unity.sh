#! /usr/bin/env bash

# This script is meant to install all the components of Shiva
​
## Setup on Ubuntu 16.04

## Installing python and venv, and creating a virtual environment

# # Installs python 3
# sudo apt install python3
# # Installs venv
# sudo apt-get install python3-venv
# cd ..
# # creates a virtual environment just for shiva
# python3 -m venv shiva
# # activates the virtual environment
# source shiva/bin/activate

## Install Unity API 

cd shiva/shiva/envs/ml-agents/ml-agents-envs/
pip install -e ./
cd ../ml-agents
pip install -e ./

# Python dependencies
pip install gym tensorboard tensorboardX tensorflow pandas torch pynput jupyter
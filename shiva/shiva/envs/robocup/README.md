# RoboCup Environment


One of the biggest components inside of Shiva is Robocup, a 2D soccer simulation to apply various machine learning algorithms.

An environment excellent for running multi agent algorithms.

## Quick Install

Simply run the following inside of this directory.

```
bash install_robocup
```

If you go back to Control-Tasks/shiva you should be run the environmen with

```
python -W ignore shiva -c DDPG-Robocup.ini -n run
```

## Manual Install

```
cd Control-Tasks/shiva/shiva/envs/robocup
git clone https://github.com/nflux/rcssserver.git
git clone https://github.com/nflux/librcsc.git
```
Whether you’re using Ubuntu, Manjaro, or MacOS, you need to install cmake, C++ Boost libraries, Flex, and Qt version 4.

### Installing RoboCup in Ubuntu

If on Ubuntu you can use apt-get package manager.
```
sudo apt-get install cmake libboost-dev libboost-all-dev flex qt4-default
```
### Installing RoboCup in Manjaro

In Manjaro you can use pacman package manager.
```
sudo pacman -S cmake boost flex qt4
```

Now that you’ve installed the libraries you should be able to build robocup servers. If you are still inside of the robocup folder then simple type the following, otherwise navigate back to the robocup folder and then proceed.
```

cd HFO
bash compile.sh
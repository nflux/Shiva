# Installing Shiva

In order to install Shiva, you have to get all parts of Shiva functional.

## Building Shiva

To build Shiva from source we first clone the repo.

> git clone https://github.com/nflux/Control-Tasks/

Then we can go into the project structure and start setting up components and dependencies. 

### Robocup

One of the biggest components inside of Shiva is Robocup, an environment excellent for running multi agent algorithms.

> cd Control-Tasks/shiva/shiva/envs/robocup
> git clone https://github.com/nflux/rcssserver.git
> git clone https://github.com/nflux/librcsc.git

Whether you’re using Ubuntu, Manjaro, or MacOS, you need to install cmake, C++ Boost libraries, Flex, and Qt version 4.

### Installing RoboCup in Ubuntu

If on Ubuntu you can use apt-get package manager.

> sudo apt-get install cmake libboost-dev libboost-all-dev flex qt4-default

### Installing RoboCup in Manjaro

In Manjaro you can use pacman package manager.

>sudo pacman -S cmake boost flex qt4

### Installing RoboCup MacOS

In MacOS you can use homebrew?

> 

Now that you’ve installed the libraries you should be able to build robocup servers. If you are still inside of the robocup folder then simple type the following, otherwise navigate back to the robocup folder and then proceed.

> cd HFO
> bash compile.sh

Lastly, you want to set up your python virtual environment. You have plenty of options to choose from (anaconda, venv, virtualenv, etc) but regardless which virtual environment manager you use there is a requirements.txt file that you can use to quickly install all the python dependencies. 

> cd ../../../../..
> pip install requirements.txt

At this point you should have shiva installed and ready to run but it would behoove you to make sure you have cuda properly installed because Shiva detects your GPU and can greatly speed up training.

## Unity

This version of Shiva does not use the gym wrapper developed by Unity's MLAgents. This version uses sockets to pass the actions to the Unity environments. Future versions of Shiva will open the scenes you create using just their binaries. editor opened with the current scene that you would like to train on.
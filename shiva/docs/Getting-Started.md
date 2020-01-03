# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

# Requirements
Must have Ubuntu 16.04
Python 3.5+

# Installing Shiva

In order to install Shiva, you have to get all parts of Shiva functional. There is a bash installation script immediately inside of the repo called install.sh. Running it will install Shiva for you given that you have Ubuntu 16.04. If, for some reason, the script fails on Ubuntu 16.04 open up an issue and we'll look into it but you can also more or less follow the instructions below and build it from source yourself.

## Building Shiva

To build Shiva from source we first clone the repo.

```
git clone https://github.com/nflux/Control-Tasks/
```

Then we can go into the project structure and start setting up components and dependencies. 


### Installing Unity API
Now, you want to set up your python virtual environment. You have plenty of options to choose from (anaconda, venv, virtualenv, etc) but regardless which virtual environment manager you use there is a requirements.txt file that you can use to quickly install all the python dependencies. To install the Unity API you have to go to Control-Tasks/shiva/shiva/envs/ml-agents/ml-agents and run

```
cd Control-Tasks/shiva/shiva/envs/ml-agents/ml-agents-envs/
pip install -e ./
cd ..
cd ml-agents/
pip install -e ./
```
Finally, in case anything was missed do pip install requirements to make sure you have the rest of the dependencies.
```
pip install requirements.txt
```
At this point you should have shiva installed and ready to run but it would behoove you to make sure you have cuda properly installed because Shiva detects your GPU and can greatly speed up training.

## Some Notes on Unity

This version of Shiva uses a gym wrapper around Unity. This version uses a Unity API to pass the actions and collect observations from the Unity environment. You only need to place your Unity binary and data files inside of unitybuilds/ directory immediately inside Shiva's environment module. More instructions can be found here [Unity Documentation](https://github.com/nflux/Control-Tasks/blob/master/shiva/docs/Unity.md) 

# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Requirements
* Ubuntu 16.04 or Mac OS
* Python 3.7
* Git
* pip (Python Package Installer)

# Installing Shiva

In order to install Shiva, you have to get all parts of Shiva functional. 

Start by cloning the repo (either thru HTTP or SSH),

```bash
# SSH
git clone git@github.com:nflux/Control-Tasks.git
# HTTP
git clone https://github.com/nflux/Control-Tasks/
```

It's highly recommended that you create a new virtual environment for the project,

```bash
python -m venv shiva_env
```

Activate the virtual environment,

```bash
source shiva_env/bin/activate
```

Then we can go into the project structure and start setting up components and dependencies. 

Run the below command in order to install the requirements.
```bash
pip install -r Control-Tasks/setup/requirements.txt
```

## Install OpenMPI

For the ability to run inter-process communication runs,
```bash
sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi1.3 libopenmpi-dbg libopenmpi-dev
```

## Install RoboCup (optional if running the soccer environment)

```bash
bash ./Control-Tasks/setup/install_robocup.sh
```

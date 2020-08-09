===================
Quick install guide
===================

Before you can use Shiva, you'll need to satisfy its requirements and install its dependencies.

Getting Started
===============

* Ubuntu 16.04 or Mac OS
* Python 3.7
* Git
* pip (Python Package Installer)

Installing Shiva
================

In order to install Shiva, you have to get all parts of Shiva functional. 

Start by cloning the repo (either thru HTTP or SSH),

.. code-block:: bash
   # SSH
   git clone git@github.com:nflux/Control-Tasks.git
   # HTTP
   git clone https://github.com/nflux/Control-Tasks/


It's highly recommended that you create a new virtual environment for the project,

.. code-block:: python

   python -m venv shiva_env


Activate the virtual environment,

.. code-block:: bash

    source shiva_env/bin/activate


Then we can go into the project structure and start setting up components and dependencies. 

Run the below command in order to install the requirements.

.. code-block:: bash

   bash ./Control-Tasks/setup/install.sh


Install OpenMPI
===============

For the ability to run inter-process communication runs,

.. code-block:: linux

   sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi1.3 libopenmpi-dbg libopenmpi-dev


Install RoboCup
===============

.. code-block:: bash

   bash ./Control-Tasks/setup/install_robocup.sh

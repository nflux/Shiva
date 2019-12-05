cd shiva/shiva/envs/robocup/

# clone the robocup files
git clone https://github.com/nflux/rcssserver.git
git clone https://github.com/nflux/librcsc.git

# Install cmake, C++ Boost Libraries, Flex and Qt version 4
sudo apt-get install cmake libboost-dev libboost-all-dev flex qt4-default
cd HFO

#The above command will create a build/ folder with various executable files
bash compile.sh
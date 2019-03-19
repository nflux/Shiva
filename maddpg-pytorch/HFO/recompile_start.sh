#!/bin/bash
rm -rf build
rm bin/rcssserver bin/soccerwindow2 hfo/libhfo_c.so
mkdir build && cd build
cmake -Wno-error=deprecated -DCMAKE_BUILD_TYPE=Debug ..
make -j4
make install

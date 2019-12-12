#!/bin/bash
rm -rf build/rcssserver-prefix
rm bin/rcssserver
# mkdir build && cd build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j8
make install

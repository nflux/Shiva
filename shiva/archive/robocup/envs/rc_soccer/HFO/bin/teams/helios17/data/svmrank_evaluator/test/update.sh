#!/bin/sh

mv /tmp/*.rank .
if [ $? -ne 0 ]; then
  exit 1
fi
cat *.rank > train.dat
svm_rank_learn -c 0.5 ../train.dat model


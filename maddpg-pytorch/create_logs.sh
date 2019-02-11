#!/bin/sh
NUM_ENVS=4
py_command=""

for ((i=1; i <= $NUM_ENVS; i++ ))
do
    j=$((($i+1) * 1000))
    if [ "$j" != "$((($NUM_ENVS+1) * 1000))" ]; then
        temp="python create_pretrain_files.py --port $j --log_dir log_$j |"
        py_command="$py_command $temp"
    else
        temp="python create_pretrain_files.py --port $j --log_dir log_$j"
        py_command="$py_command $temp"
    fi
done

eval $py_command
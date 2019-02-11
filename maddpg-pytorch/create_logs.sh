#!/bin/sh
NUM_ENVS=3
py_command=""

for ((i=1; i <= $NUM_ENVS; i++ ))
do
    j=$((($i+1) * 1000))
    if ["$j" != "$((($NUM_ENVS+1) * 1000))"]
    then
        $py_command="python create_pretrain_files.py --port $j --log_dir log_$j | ${py_command}"
    else
        $py_command="${py_command}python create_pretrain_files.py --port $j --log_dir log_$j"
    fi
done

eval $py_command
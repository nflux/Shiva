#!/bin/sh
NUM_ENVS=8
py_command=""

declare -a arr_offense_teams=("helios18" "helios18" "helios18" "helios18" "helios18" "helios18" "helios18" "helios18")
declare -a arr_defense_teams=("helios11" "helios12" "helios" "cyrus" "helios15" "helios16" "helios17" "helios18")

for ((i=1; i <= $NUM_ENVS; i++ ))
do
    j=$((($i+1) * 1000))
    if [ "$j" != "$((($NUM_ENVS+1) * 1000))" ]; then
        temp="python create_pretrain_files.py --port $j --log_dir log_$j --offense-team ${arr_offense_teams[$i-1]} --defense-team ${arr_defense_teams[$i-1]} |"
        py_command="$py_command $temp"
    else
        temp="python create_pretrain_files.py --port $j --log_dir log_$j --offense-team ${arr_offense_teams[$i-1]} --defense-team ${arr_defense_teams[$i-1]}"
        py_command="$py_command $temp"
    fi
done

eval $py_command
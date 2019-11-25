#!/bin/sh

echo "******************************************************************"
echo " HELIOS2014"
echo " National Institute of Advanced Industrial Science and Technology"
echo " Created by Hidehisa Akiyama and Hiroki Shimora"
echo " Copyright 2000-2007.  Hidehisa Akiyama"
echo " Copyright 2007-2010.  Hidehisa Akiyama and Hiroki Shimora"
echo " Copyright 2011- Hidehisa Akiyama, Hiroki Shimora,"
echo "   Tomoharu Nakashima (2011-),"
echo "   Yousuke Narimoto, Tomohiko Okayama (2011-)"
echo "   Katsuhiro Yamashita (2013-)"
echo "   Satoshi Mifune (2014-)"
echo " All rights reserved."
echo "******************************************************************"

DIR=`dirname $0`
echo $DIR
LIBPATH=$DIR/local/lib
if [ x"$LIBPATH" != x ]; then
  if [ x"$LD_LIBRARY_PATH" = x ]; then
    LD_LIBRARY_PATH=$LIBPATH
  else
    LD_LIBRARY_PATH=$LIBPATH:$LD_LIBRARY_PATH
  fi
  export LD_LIBRARY_PATH
fi

player="${DIR}/helios_player"
coach="${DIR}/helios_coach"
teamname="HELIOS2014"
host="localhost"
port=6000
coach_port=6002

player_conf="${DIR}/player.conf"
formation_dir="${DIR}/data/formations"
formation_conf="${DIR}/data/formation.conf"
overwrite_formation_conf="${DIR}/data/overwrite_formation.conf"
hetero_conf="${DIR}/data/hetero.conf"
ball_table_file="${DIR}/data/ball_table.dat"

goalie_position_dir="${DIR}/data/goalie_position/"
intercept_conf_dir="${DIR}/data/intercept_probability/"
opponent_data_dir="${DIR}/data/opponent_data/"

chain_search_method="BestFirstSearch"
evaluator_name="Default"
sirm_evaluator_param_dir="${DIR}/data/sirm_evaluator/"
max_chain_length="4"
max_evaluate_size="1000"

coach_conf="${DIR}/coach.conf"
team_graphic="--use_team_graphic on"

ping -c 1 $host

common_opt=""
common_opt="${common_opt} -h ${host} -t ${teamname}"
common_opt="${common_opt} --formation-conf-dir ${formation_dir}"
common_opt="${common_opt} --formation-conf ${formation_conf}"
common_opt="${common_opt} --overwrite-formation-conf ${overwrite_formation_conf}"
common_opt="${common_opt} --hetero-conf ${hetero_conf}"
common_opt="${common_opt} --ball-table ${ball_table_file}"
common_opt="${common_opt} --chain-search-method ${chain_search_method}"
common_opt="${common_opt} --evaluator-name ${evaluator_name}"
common_opt="${common_opt} --max-chain-length ${max_chain_length}"
common_opt="${common_opt} --max-evaluate-size ${max_evaluate_size}"
common_opt="${common_opt} --sirm-evaluator-param-dir ${sirm_evaluator_param_dir}"
common_opt="${common_opt} --goalie-position-dir ${goalie_position_dir}"
common_opt="${common_opt} --intercept-conf-dir ${intercept_conf_dir}"
common_opt="${common_opt} --opponent-data-dir ${opponent_data_dir}"

player_opt="--player-config ${player_conf}"
player_opt="${player_opt} ${common_opt}"
player_opt="${player_opt} -p ${port}"

coach_opt="--coach-config ${coach_conf}"
coach_opt="${coach_opt} ${common_opt}"
coach_opt="${coach_opt} -p ${coach_port}"
coach_opt="${coach_opt} ${team_graphic}"

echo "player options: $player_opt"
echo "coach options: $coach_opt"


$player ${player_opt} -g &
sleep 1
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
$player ${player_opt} &
sleep 1
$coach ${coach_opt} &


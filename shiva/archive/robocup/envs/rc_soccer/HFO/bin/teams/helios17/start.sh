#!/bin/sh

echo "******************************************************************"
echo " HELIOS2017"
echo " Fukuoka University & Osaka Prefecture University"
echo " Copyright 2000-2007.  Hidehisa Akiyama"
echo " Copyright 2007-2010.  Hidehisa Akiyama and Hiroki Shimora"
echo " Copyright 2011- Hidehisa Akiyama, Hiroki Shimora,"
echo "   Tomoharu Nakashima (2011-),"
echo "   Yousuke Narimoto, Tomohiko Okayama (2011-)"
echo "   Katsuhiro Yamashita (2013-)"
echo "   Satoshi Mifune (2014-)"
echo "   Sho Tanaka, Jordan Henrio (2015-)"
echo "   Tomonari Nakade, Takuya Fukushima (2016-)"
echo "   Yudai Suzuki, An Ohori (2017-)"
echo " All rights reserved."
echo ""
echo " LIBLINEAR"
echo "   Copyright (c) 2007-2015 The LIBLINEAR Project."
echo "   All rights reserved."
echo " LIBSVM"
echo "   Copyright (c) 2000-2014 Chih-Chung Chang and Chih-Jen Lin"
echo "   All rights reserved."
echo "******************************************************************"

DIR=`dirname $0`

LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

player="${DIR}/helios_player"
coach="${DIR}/helios_coach"
teamname="HELIOS2017"
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
svmrank_evaluator_model="${DIR}/data/svmrank_evaluator/model"
neural_network_evaluator_dir="${DIR}/data/neural_network_evaluator/"
center_forward_free_move_model="${DIR}/data/center_forward_free_move/model"
svm_formation_classifier_model="${DIR}/data/svm_formation_classifier/svm.model"
max_chain_length="4"
max_evaluate_size="1000"

coach_conf="${DIR}/coach.conf"
team_graphic="--use_team_graphic on"

sleepprog=sleep
goaliesleep=1
sleeptime=0



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
common_opt="${common_opt} --svmrank-evaluator-model ${svmrank_evaluator_model}"
common_opt="${common_opt} --neural-network-evaluator-dir ${neural_network_evaluator_dir}"
common_opt="${common_opt} --center-forward-free-move-model ${center_forward_free_move_model}"
common_opt="${common_opt} --svm-formation-classifier-model ${svm_formation_classifier_model}"
common_opt="${common_opt} --goalie-position-dir ${goalie_position_dir}"
common_opt="${common_opt} --intercept-conf-dir ${intercept_conf_dir}"
common_opt="${common_opt} --opponent-data-dir ${opponent_data_dir}"
common_opt="${common_opt} --debug_server_host ${debug_server_host}"
common_opt="${common_opt} --debug_server_port ${debug_server_port}"
common_opt="${common_opt} ${offline_logging}"
common_opt="${common_opt} ${debug_opt}"

player_opt="--player-config ${player_conf}"
player_opt="${player_opt} ${common_opt}"
player_opt="${player_opt} -p ${port}"
player_opt="${player_opt} ${fullstate_opt}"

coach_opt="--coach-config ${coach_conf}"
coach_opt="${coach_opt} ${common_opt}"
coach_opt="${coach_opt} -p ${coach_port}"
coach_opt="${coach_opt} ${team_graphic}"

echo "player options: $player_opt"
echo "coach options: $coach_opt"


$player ${player_opt} -g ${offline_number} &
$sleepprog $goaliesleep

i=2
while [ $i -le 11 ] ; do
  $player ${player_opt} ${offline_number} &
  $sleepprog $sleeptime
  i=`expr $i + 1`
done

$coach ${coach_opt} ${offline_mode} &

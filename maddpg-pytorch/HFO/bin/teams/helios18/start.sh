#!/bin/sh

echo "******************************************************************"
echo " HELIOS2018"
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

LIBPATH=$DIR/lib
if [ x"$LIBPATH" != x ]; then
  if [ x"$LD_LIBRARY_PATH" = x ]; then
    LD_LIBRARY_PATH=$LIBPATH
  else
    LD_LIBRARY_PATH=$LIBPATH:$LD_LIBRARY_PATH
  fi
  export LD_LIBRARY_PATH
fi

DIR=`dirname $0`

echo "I have reached the start.h file for Helios 18"

player="${DIR}/helios_player"
coach="${DIR}/helios_coach"
teamname="HELIOS2018"
host="localhost"
port=6000
coach_port=""
debug_server_host=""
debug_server_port=""

player_conf="${DIR}/player.conf"
formation_dir="${DIR}/data/formations"
setplay_dir="${DIR}/data/setplay"
formation_conf="${DIR}/data/formation.conf"
overwrite_formation_conf="${DIR}/data/overwrite_formation.conf"
hetero_conf="${DIR}/data/hetero.conf"

intercept_conf_dir="${DIR}/data/intercept_probability/"
opponent_data_dir="${DIR}/data/opponent_data/"

chain_search_method="BestFirstSearch"
evaluator_name="Default"
intercept_evaluator_name="Default"
svmrank_evaluator_model="${DIR}/data/svmrank_evaluator/model"
svmrank_intercept_evaluator_model="${DIR}/data/svmrank_intercept_evaluator/model"
neural_network_evaluator_dir="${DIR}/data/neural_network_evaluator/"
center_forward_free_move_model="${DIR}/data/center_forward_free_move/model"
svm_formation_classifier_model="${DIR}/data/svm_formation_classifier/svm.model"
max_chain_length="3"
max_evaluate_size="3000"

coach_conf="${DIR}/coach.conf"
team_graphic="--use_team_graphic off"

audio_shift="0"

number=11
usecoach="true"

unum=0

sleepprog=sleep
goaliesleep=1
sleeptime=0

debug_opt=""

offline_logging=""
offline_mode=""
fullstate_opt=""

foreground="false"

usage()
{
  (echo "Usage: $0 [options]"
   echo "Available options:"
   echo "      --help                   prints this"
   echo "  -h, --host HOST              specifies server host (default: localhost)"
   echo "  -p, --port PORT              specifies server port (default: 6000)"
   echo "  -P  --coach-port PORT        specifies server port for online coach (default: 6002)"
   echo "  -t, --teamname TEAMNAME      specifies team name"
   echo "  -n, --number NUMBER          specifies the number of players"
   echo "  -u, --unum UNUM              specifies the invoked player/coach by uniform"
   echo "  -C, --without-coach          specifies not to run the coach"
   echo "  -f, --formation DIR          specifies the formation directory"
   echo "      --setplay-dir DIR        specifies the setplay directory"
   echo "      --chain-search-method NAME specifies the search algorithm {BestFirstSearch|MonteCarloTreeSearch}"
   echo "      --evaluator-name NAME    specifies the field evaluator"
   echo "      --intercept-evaluator-name NAME    specifies the intercept evaluator"
   echo "      --max-chain-length N     specifies the maximum action chain length"
   echo "      --max-evaluate-size N    specifies the maximum action chain size to be evaluated"
   echo "      --intercept-conf-dir DIR specifies the directory path for intercept conf files"
   echo "      --opponent-data-dir  DIR specifies the directory path for analyzed opponent data files"
   echo "  --team-graphic FILE          specifies the team graphic xpm file"
   echo "  --foreground                 wait child precesses") 1>&2
}

while [ $# -gt 0 ]
do
  case $1 in

    --help)
      usage
      exit 0
      ;;

    -h|--host)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      host="${2}"
      shift 1
      ;;

    -p|--port)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      port="${2}"
      shift 1
      ;;

    -P|--coach-port)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      coach_port="${2}"
      shift 1
      ;;

    -t|--teamname)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      teamname="${2}"
      shift 1
      ;;

    -n|--number)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      number="${2}"
      shift 1
      ;;

    -u|--unum)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      unum="${2}"
      shift 1
      ;;

    -C|--without-coach)
      usecoach="false"
      ;;

    -f|--formation)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      formation_dir="${2}"
      shift 1
      ;;

    --setplay-dir)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      setplay_dir="${2}"
      shift 1
      ;;

    --chain-search-method)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      chain_search_method="${2}"
      shift 1
      ;;

    --evaluator-name)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      evaluator_name="${2}"
      shift 1
      ;;

    --intercept-evaluator-name)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      intercept_evaluator_name="${2}"
      shift 1
      ;;

    --max-chain-length)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      max_chain_length="${2}"
      shift 1
      ;;

    --max-evaluate-size)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      max_evaluate_size="${2}"
      shift 1
      ;;

    --intercept-conf-dir)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      intercept_conf_dir="${2}"
      shift 1
      ;;

    --opponent-data-dir)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      opponent_data_dir="${2}"
      shift 1
      ;;

    --team-graphic)
      if [ $# -lt 2 ]; then
        usage
        exit 1
      fi
      team_graphic="--use_team_graphic on --team_graphic_file ${2}"
      shift 1
      ;;

    --foreground)
      foreground="true"
      ;;

    *)
      echo 1>&2
      echo "invalid option \"${1}\"." 1>&2
      echo 1>&2
      usage
      exit 1
      ;;
  esac

  shift 1
done

if [ X"${coach_port}" = X'' ]; then
  coach_port=`expr ${port} + 2`
fi

ping -c 1 $host

common_opt=""
common_opt="${common_opt} -h ${host} -t ${teamname}"
common_opt="${common_opt} --formation-dir ${formation_dir}"
common_opt="${common_opt} --setplay-dir ${setplay_dir}"
common_opt="${common_opt} --formation-conf ${formation_conf}"
common_opt="${common_opt} --overwrite-formation-conf ${overwrite_formation_conf}"
common_opt="${common_opt} --hetero-conf ${hetero_conf}"
common_opt="${common_opt} --chain-search-method ${chain_search_method}"
common_opt="${common_opt} --evaluator-name ${evaluator_name}"
common_opt="${common_opt} --intercept-evaluator-name ${intercept_evaluator_name}"
common_opt="${common_opt} --max-chain-length ${max_chain_length}"
common_opt="${common_opt} --max-evaluate-size ${max_evaluate_size}"
common_opt="${common_opt} --svmrank-evaluator-model ${svmrank_evaluator_model}"
common_opt="${common_opt} --svmrank-intercept-evaluator-model ${svmrank_intercept_evaluator_model}"
common_opt="${common_opt} --neural-network-evaluator-dir ${neural_network_evaluator_dir}"
common_opt="${common_opt} --center-forward-free-move-model ${center_forward_free_move_model}"
common_opt="${common_opt} --svm-formation-classifier-model ${svm_formation_classifier_model}"
common_opt="${common_opt} --intercept-conf-dir ${intercept_conf_dir}"
common_opt="${common_opt} --opponent-data-dir ${opponent_data_dir}"
common_opt="${common_opt} --debug_server_host ${debug_server_host}"
common_opt="${common_opt} --debug_server_port ${debug_server_port}"
common_opt="${common_opt} --audio_shift ${audio_shift}"
common_opt="${common_opt} --record"

player_opt="--player-config ${player_conf}"
player_opt="${player_opt} ${common_opt}"
player_opt="${player_opt} -p ${port}"

coach_opt="--coach-config ${coach_conf}"
coach_opt="${coach_opt} ${common_opt}"
coach_opt="${coach_opt} -p ${coach_port}"
coach_opt="${coach_opt} ${team_graphic}"

echo "player options: $player_opt"
echo "coach options: $coach_opt"


if [ $number -gt 0 ]; then
  if [ $unum -eq 0 -o $unum -eq 1 ]; then
    $player ${player_opt} -g &
    $sleepprog $goaliesleep
  fi
fi

i=2
while [ $i -le ${number} ] ; do
  if [ $unum -eq 0 -o $unum -eq $i ]; then
    $player ${player_opt} &
    $sleepprog $sleeptime
  fi

  i=`expr $i + 1`
done

if [ "${usecoach}" = "true" ]; then
  if [ $unum -eq 0 -o $unum -eq 12 ]; then
    $coach ${coach_opt} &
  fi
fi

if [ "${foreground}" = "true" ]; then
  wait
fi

#!/bin/sh

# HOST=$1
# BASEDIR=$2
# NUM=$3

# HOST="localhost"
# port=6000
# coach_port=6002

# LIBPATH=./lib
# if [ x"$LIBPATH" != x ]; then
#   if [ x"$LD_LIBRARY_PATH" = x ]; then
#     LD_LIBRARY_PATH=$LIBPATH
#   else
#     LD_LIBRARY_PATH=$LIBPATH:$LD_LIBRARY_PATH
#   fi
#   export LD_LIBRARY_PATH
# fi

# teamname="CYRUS2018"

# player="./src/sample_player"
# coach="./src/sample_coach"

# config="src/player.conf"
# config_dir="src/formations-dt"
# coach_config="src/coach.conf"

# opt="--player-config ${config} --config_dir ${config_dir}"
# opt="${opt} -h ${HOST} -t ${teamname}"

# coachopt="--coach-config ${coach_config}"
# coachopt="${coachopt} -h ${HOST} -t ${teamname}"

# cd $BASEDIR

# case $NUM in
#     1)
#         $player $opt -g
#         ;;
#     12)
#         $coach $coachopt
#         ;;
#     *)
#         $player $opt
#         ;;
# esac

#!/bin/sh

DIR=`dirname $0`
player="${DIR}/sample_player"
coach="${DIR}src/sample_coach"
teamname="Cyrus2018"
host="localhost"
port=6000
coach_port=6002

player_conf="${DIR}/player.conf"
formation_dir="${DIR}/src/formations-dt"


coach_conf="${DIR}/coach.conf"


usage()
{
  (echo "Usage: $0 [options]"
   echo "Available options:"
   echo "      --help                   prints this"
   echo "  -h, --host HOST              specifies server host (default: localhost)"
   echo "  -p, --port PORT              specifies server port (default: 6000)"
   echo "  -P, --coach-port PORT        specifies server port for online coach (default: 6002)") 1>&2
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

ping -c 1 $host

common_opt=""
common_opt="${common_opt} -h ${host} -t ${teamname}"
common_opt="${common_opt} --formation-conf-dir ${formation_dir}"
# common_opt="${common_opt} --role-conf ${role_conf}"
# common_opt="${common_opt} --ball-table ${ball_table_file}"
# common_opt="${common_opt} --chain-search-method ${chain_search_method}"
# common_opt="${common_opt} --evaluator-name ${evaluator_name}"
# common_opt="${common_opt} --max-chain-length ${max_chain_length}"
# common_opt="${common_opt} --max-evaluate-size ${max_evaluate_size}"
# common_opt="${common_opt} --sirm-evaluator-param-dir ${sirm_evaluator_param_dir}"
# common_opt="${common_opt} --goalie-position-dir ${goalie_position_dir}"
#common_opt="${common_opt} --intercept-conf-dir ${intercept_conf_dir}"
#common_opt="${common_opt} --opponent-data-dir ${opponent_data_dir}"

player_opt="--player-config ${player_conf}"
player_opt="${player_opt} ${common_opt}"
player_opt="${player_opt} -p ${port}"

coach_opt="--coach-config ${coach_conf}"
coach_opt="${coach_opt} ${common_opt}"
coach_opt="${coach_opt} -p ${coach_port}"


$player ${player_opt} -g &
#$sleepprog $goaliesleep

i=2
while [ $i -le 11 ] ; do
  $player ${player_opt} &
  #$sleepprog $playersleep

  i=`expr $i + 1`
done

#$sleepprog $coachsleep
$coach ${coach_opt} &
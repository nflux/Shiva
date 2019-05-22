Installation instructions:

mkdir shiva
cd shiva

Get the RCSSSERVER Repo:

git clone https://github.com/mehrzadshabez/rcssserver.git
cd rcssserver
cd ..

Get the LIBRCSC Repo:

git clone https://github.com/mehrzadshabez/librcsc.git
cd librcsc
cd ..


Get the Robocup-Sigma Repo:

git clone https://github.com/mehrzadshabez/Robocup-Sigma.git
cd Robocup-Sigma

cd envs/rc_soccer/HFO
bash recompile_start.sh



Training instructions:

from shiva/Robocup-Sigma run

python main.py

--env options: {rc,nmmo}
--conf options {rc_cpu,rc_gpu,rc_multi_gpu}
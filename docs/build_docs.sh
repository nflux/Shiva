
make clean

# PYTHONPATH=. sphinx-apidoc    -f -o source/rst/ ../shiva/shiva/
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/algorithms       ../shiva/shiva/algorithms
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/buffers          ../shiva/shiva/buffers
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/learners         ../shiva/shiva/learners
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/envs             ../shiva/shiva/envs
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/agents           ../shiva/shiva/agents
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/networks         ../shiva/shiva/networks
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/eval_envs        ../shiva/shiva/eval_envs
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/core             ../shiva/shiva/core
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/helpers          ../shiva/shiva/helpers
PYTHONPATH=. sphinx-apidoc -e -f -o source/rst/helpers/utils    ../shiva/shiva/helpers/utils

make html
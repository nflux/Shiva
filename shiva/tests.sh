#!/bin/bash
FILENAME=/tests/logs/$(date +"%Y-%b-%d-%Z-%H-%m-%S")
mkdir -m 777 -p $FILENAME
#pytest tests --log-file $FILENAME
pytest tests
#!/bin/sh

cd /home/guilherme/Documents/noa/cidia19/pipeline-app-3

eval "$(conda shell.bash hook)" && conda activate py36 && echo py36 && python pipeline.py && conda deactivate

echo "finished"

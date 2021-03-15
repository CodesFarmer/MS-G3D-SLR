#!/bin/bash
export PYTHON_HOME=/root/paddlejob/miniconda3
export PATH=${PYTHON_HOME}/bin:$PATH
export PYTHONPATH=${PYTHON_HOME}/lib/python3.7:$PYTHONPATH
export PYTHONPATH=${PYTHON_HOME}/lib/python3.7/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:$LD_LIBRARY_PATH 

python gen_bone_data.py --dataset hand

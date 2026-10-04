#!/bin/bash

#PBS -N RSF_C+E+N+G+F+T+S_Both
#PBS -P 11004309
#PBS -j oe
#PBS -o /home/users/nus/e1083772/cancer-survival-ml/.pbs/3_gridsearchcv/rsf/
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=32G
#PBS -l walltime=04:00:01

set -e

module load python/3.12.1-gcc11

source /home/users/nus/e1083772/python3.12_venv/bin/activate

python /home/users/nus/e1083772/cancer-survival-ml/pipeline/3_gridsearchcv.py --architecture RSF --n-iter 100 --endpoint both
#!/bin/bash

#PBS -N naive0imp_mut_os
#PBS -P 11004309
#PBS -j oe
#PBS -o /home/users/nus/e1083772/cancer-survival-ml/.pbs/3_gridsearchcv/vae_zero_impute_naive/
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=32G
#PBS -l walltime=8:00:01

set -e

module load python/3.12.1-gcc11

source /home/users/nus/e1083772/python3.12_venv/bin/activate

python /home/users/nus/e1083772/cancer-survival-ml/pipeline/3_gridsearchcv.py --n-iter 100 --endpoint os
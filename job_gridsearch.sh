#!/bin/bash

#PBS -N naive0imp_C+E+N+G+F+T+M_pfs
#PBS -P 11004309
#PBS -j oe
#PBS -o /home/users/nus/e1083772/cancer-survival-ml/.pbs/3_gridsearchcv/vae_zero_impute_naive/
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=32G
#PBS -l walltime=4:00:01
#PBS -J 0-49

set -e

module load python/3.12.1-gcc11

source /home/users/nus/e1083772/python3.12_venv/bin/activate

python /home/users/nus/e1083772/cancer-survival-ml/pipeline/3_gridsearchcv.py --n-iter 100 --endpoint pfs
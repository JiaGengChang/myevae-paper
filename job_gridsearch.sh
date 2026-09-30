#!/bin/bash

#PBS -N vae_mutation
#PBS -P 11004309
#PBS -j oe
#PBS -o /home/users/nus/e1083772/cancer-survival-ml/.pbs/3_gridsearchcv/vae_mutation/
#PBS -q normal
#PBS -l select=1:ncpus=2:mem=16G
#PBS -l walltime=00:20:00
#PBS -J 1-49

set -e

module load python/3.12.1-gcc11

source /home/users/nus/e1083772/python3.12_venv/bin/activate

python /home/users/nus/e1083772/cancer-survival-ml/pipeline/3_gridsearchcv.py --n-iter 10 --endpoint os
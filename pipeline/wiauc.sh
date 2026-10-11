#!/bin/bash

#PBS -N score_wiauc
#PBS -P 11004309
#PBS -j oe
#PBS -o /home/users/nus/e1083772/cancer-survival-ml/.pbs/4_score_wiauc/
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=8G
#PBS -l walltime=00:59:01

module load python/3.12.1-gcc11

source /home/users/nus/e1083772/python3.12_venv/bin/activate

cd /home/users/nus/e1083772/cancer-survival-ml/;

for shuffle in {0..9}; do 
    for fold in {0..4}; do  
        python pipeline/4_score_wiauc.py output/rsf_models/exp-cna/pfs_shuffle"${shuffle}"_fold"${fold}".json;
    done;
done


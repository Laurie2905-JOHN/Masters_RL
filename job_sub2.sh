#!/bin/bash
#PBS -l select=1:ncpus=32:mem=62gb:ngpus=0
#PBS -l walltime=16:00:00
#PBS -N tune6
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/tune6.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/tune6_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/hyperparam_search/optuna_search_js.py" \
    --algo="A2C" \
    --study_name="A2C_study_newR_j" \
    --n_trials=1000 \
    --timeout=86500 \
    --n_jobs=8 \
    --num_timesteps=1000000

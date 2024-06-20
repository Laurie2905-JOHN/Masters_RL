#!/bin/bash
#PBS -l select=1:ncpus=32:mem=62gb:ngpus=0
#PBS -l walltime=24:00:00
#PBS -N optuna_training
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/A2C_opt1.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/A2C_opt1_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/hyperparam_search/optuna_search.py" \
    --algo="A2C" \
    --study_name="A2C_study" \
    --n_trials=1000 \
    --timeout=86500 \
    --n_jobs=32 \
    --num_timesteps=1000000

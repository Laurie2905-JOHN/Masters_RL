#!/bin/bash
#PBS -l select=1:ncpus=16:mem=16gb:ngpus=0
#PBS -l walltime=05:00:00
#PBS -N A2C_benchmark
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/A2C_benchmark.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/A2C_benchmark_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/hyperparam_search/optuna_search.py" \
    --algo="A2C" \
    --study_name="A2C_benchmark" \
    --n_trials=8 \
    --timeout=14400 \
    --num_timesteps=20000

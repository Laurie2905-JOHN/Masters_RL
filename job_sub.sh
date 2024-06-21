#!/bin/bash
#PBS -l select=1:ncpus=32:mem=64gb:ngpus=0
#PBS -l walltime=24:00:00
#PBS -N parralel_optuna13
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/parralel_optuna13.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/parralel_optuna_error13.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

  python "scripts/training/hyperparam_search/optuna_search_js.py" \
    --algo="A2C" \




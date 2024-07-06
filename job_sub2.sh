#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32gb:ngpus=0
#PBS -l walltime=12:00:00
#PBS -N run_multiple_configs
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/run_multiple_configs.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/run_multiple_configs_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/run_multiple_configs.py" \

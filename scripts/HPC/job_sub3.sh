#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32GB:ngpus=1
#PBS -l walltime=48:00:00
#PBS -N menu_gen_update
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/menu_gen_update.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/menu_gen_update_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/preference/report_tests/menu_generation_test.py"

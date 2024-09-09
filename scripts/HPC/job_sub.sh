#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32GB:ngpus=0
#PBS -l walltime=48:00:00
#PBS -N feedback_mech_test
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_mech_test.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_mech_test_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/preference/report_tests/feedback_mechanism_test.py"

# Run 20
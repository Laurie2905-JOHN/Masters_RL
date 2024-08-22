#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32GB:ngpus=0
#PBS -l walltime=34:30:00
#PBS -N feedback_reward_test2
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_reward_test3.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_reward_test_error3.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/preference/report_tests/feedback_reward_test.py"

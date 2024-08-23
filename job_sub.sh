#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32GB:ngpus=1
#PBS -l walltime=20:00:00
#PBS -N feedback_reward_test_final_RL_methods
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_reward_test_final_RL_methods.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_reward_test_final_RL_methods_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/preference/report_tests/feedback_reward_test.py"

# RUn 13 for prob methods starting at 50\% 
# Run 14 for genetic method starting at 50\% only doing 50
# RUn 15 for RL method starting at 50\% only doing 50
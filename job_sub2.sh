#!/bin/bash
#PBS -l select=1:ncpus=32:mem=32GB:ngpus=0
#PBS -l walltime=30:00:00
#PBS -N feedback_mech_test2
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_mech_test_final_prob_methods2.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/feedback_mech_test_final_prob_methods_error2.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/preference/report_tests/feedback_mechanism_test.py"

#run11, only for probs methods run12 will be for genetic which will start from an initial split of 50\% RL will be run seperate and will also do this
# Will only be ran for the perfect methods others this will be for all. This starts from 0.05 split

# ^ This might have changed as new run run 13 other methods run 12 for genetic keeping both running
#!/bin/bash
#PBS -l select=1:ncpus=10:mem=24gb:ngpus=1
#PBS -l walltime=10:00:00
#PBS -N RL_Algo_Test
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/algo_test_env2_1000step.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/algo_test_env2_1000step.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

# Run the Python script
python "/rds/general/user/lej23/home/fyp/Masters_RL/scripts/training/train2.py"

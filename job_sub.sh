#!/bin/bash
#PBS -l select=1:ncpus=10:mem=16gb:ngpus=0
#PBS -l walltime=4:00:00
#PBS -N RL_Algo_Test
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/algo_test_A2C_1mill.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/algo_test_A2C_1mill_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

# Run the Python script with A2C algorithm
python "/rds/general/user/lej23/home/fyp/Masters_RL/scripts/training/train.py" --algo A2C --env_name SchoolMealSelection-v0 --num_envs 4 --total_timesteps 1000000 --save_freq 10000 --eval_freq 100
# Run the Python script with PPO algorithm
# python "/rds/general/user/lej23/home/fyp/Masters_RL/scripts/training/train.py" --algo PPO --env_name SchoolMealSelection-v0 --num_envs 4 --total_timesteps 100000 --save_freq 10000 --eval_freq 100 

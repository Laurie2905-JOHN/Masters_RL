#!/bin/bash
#PBS -l select=1:ncpus=10:mem=64gb:ngpus=0
#PBS -l walltime=15:00:00
#PBS -N RL_Algo_Test_GPU
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=3 \
  --total_timesteps=3000000 \
  --save_freq=1000 \
  --eval_freq=1000 \
  --device="cuda" \
#!/bin/bash
#PBS -l select=1:ncpus=16:mem=16gb:ngpus=0
#PBS -l walltime=3:00:00
#PBS -N New_Reward3
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/train_hyperparam_newReward.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/train_hyperparam_newReward_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

# Execute the Python script with the provided arguments
python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=16 \
  --total_timesteps=1000000 \
  --save_freq=10000 \
  --eval_freq=1000 \
  --device="cpu" \
  --memory_monitor=False \
  --plot_reward_history=True \
  --log_prefix="1000_newReward"
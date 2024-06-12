#!/bin/bash
#PBS -l select=1:ncpus=10:mem=64gb:ngpus=0
#PBS -l walltime=10:00:00
#PBS -N RL_Algo_Test_CPU_seed2
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test1.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test_error1.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=3 \
  --total_timesteps=4000000 \
  --save_freq=100000 \
  --eval_freq=10000 \
  --device="cpu" \
  --memory_monitor="True" \
  --reward_metrics="['nutrients', 'groups', 'environment', 'cost', 'consumption]" \
  --plot_reward_history=False \
  # --seed="[1610340177]"
  # --checkpoint_path="saved_models/checkpoints/SchoolMealSelection_v1_A2C_10000_1env_nutrients_groups_environment_seed_1610340177_15000_steps" \
  
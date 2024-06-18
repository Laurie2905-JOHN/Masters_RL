#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb:ngpus=0
#PBS -l walltime=8:00:00
#PBS -N 100step
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2_100step.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2_100step_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

    python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=1000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --max_episode_steps=100 \
    --reward_metrics="nutrients,groups,environment,cost,consumption" \
    --log_prefix="100_step"
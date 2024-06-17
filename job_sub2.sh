#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb:ngpus=0
#PBS -l walltime=8:00:00
#PBS -N test2
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/test2_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

    python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="nutrients" \

    python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="nutrients,environment,cost,consumption" \

    python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="groups,environment,cost,consumption" \
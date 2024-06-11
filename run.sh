#!/bin/bash

python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=1 \
  --total_timesteps=10000 \
  --save_freq=10000 \
  --eval_freq=1000 \
  --device="cuda" \
  --checkpoint_path="saved_models/checkpoints/SchoolMealSelection_v1_A2C_10000_1env_nutrients_groups_environment_seed_1610340177_10000_steps" \
  --seed="[1610340177]"




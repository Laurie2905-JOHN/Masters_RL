#!/bin/bash

python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=1 \
  --total_timesteps=20000 \
  --save_freq=1000 \
  --eval_freq=1000 \
  --device="cpu" \
  --memory_monitor="True" \
  --reward_metrics="['nutrients', 'groups', 'environment', 'cost', 'consumption]" \
  # --seed="[1610340177]"
  # --checkpoint_path="saved_models/checkpoints/SchoolMealSelection_v1_A2C_10000_1env_nutrients_groups_environment_seed_1610340177_15000_steps" \
  




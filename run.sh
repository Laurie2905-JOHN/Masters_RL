#!/bin/bash

python "scripts/training/train.py" \
  --algo="A2C" \
  --env_name="SchoolMealSelection-v1" \
  --num_envs=1 \
  --total_timesteps=2000000 \
  --save_freq=10000 \
  --eval_freq=1000 \
  --device="cuda" \
  --log_prefix="test_gpu"



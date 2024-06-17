#!/bin/bash

# # List of reward metrics to test
# metrics=("nutrients" "groups" "environment" "cost" "consumption")

# # Function to join array elements with a comma
# join_by() {
#   local IFS="$1"
#   shift
#   echo "$*"
# }

# # Loop through each combination of metrics
# for i in ${!metrics[@]}; do
#   current_metrics=(${metrics[@]:0:$(($i + 1))})
#   metrics_str=$(join_by , "${current_metrics[@]}")
#   prefix_str=$(join_by _ "${current_metrics[@]}")

#   echo "Testing with reward metrics: $metrics_str"

#   python "scripts/training/train.py" \
#     --algo="A2C" \
#     --env_name="SchoolMealSelection-v1" \
#     --num_envs=16 \
#     --total_timesteps=2000000 \
#     --save_freq=1000 \
#     --eval_freq=1000 \
#     --device="cpu" \
#     --memory_monitor="False" \
#     --reward_metrics="$metrics_str" \
#     --save_prefix="$prefix_str" \
#     --log_prefix="$prefix_str" \
#     --best_prefix="$prefix_str" \
#     --reward_prefix="$prefix_str"
# done


  python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="nutrients,groups" \

  python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="nutrients,groups" \

  python "scripts/training/train.py" \
    --algo="A2C" \
    --env_name="SchoolMealSelection-v1" \
    --num_envs=16 \
    --total_timesteps=3000000 \
    --save_freq=10000 \
    --eval_freq=1000 \
    --device="cpu" \
    --memory_monitor="False" \
    --reward_metrics="groups" \

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
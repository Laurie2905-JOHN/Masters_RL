#!/bin/bash

# Arrays of different settings
ALGOS=("PPO" "A2C")

# Number of runs
# NUM_RUNS=${#ALGOS[@]}

# Loop to run the command with different settings
# for ((i=0; i<NUM_RUNS; i++))
# do
# echo "Running iteration $((i+1)) with algo=${ALGOS[i]}"
python scripts/training/train.py --algo="A2C" --env_name="CalorieOnlyEnv-v3" --num_envs=3 --total_timesteps=500000 --save_freq=10000 --eval_freq=1000 --device=cuda --seed=89 --num_people=50 --save_prefix="new_reward_test" --log_prefix="new_reward_test"
# python scripts/training/train.py --algo="A2C" --env_name="CalorieOnlyEnv-v3" --num_envs=3 --total_timesteps=2500000 --save_freq=10000 --eval_freq=1000 --device=cuda --seed=89 --num_people=200 --save_prefix="200_people" --log_prefix="200_people"
# python scripts/training/train.py --algo="A2C" --env_name="CalorieOnlyEnv-v3" --num_envs=3 --total_timesteps=2500000 --save_freq=10000 --eval_freq=1000 --device=cuda --seed=89 --num_people=100 --save_prefix="100_people" --log_prefix="100_people"
# done

# echo "Completed $NUM_RUNS runs."
#  python scripts/training/train.py --algo=${ALGOS[i]} --env_name="CalorieOnlyEnv-v3" --num_envs=3 --total_timesteps=2500000 --save_freq=10000 --eval_freq=1000 --device=cuda --seed=89 ----num_people=200 --save_prefix="200_people"
 
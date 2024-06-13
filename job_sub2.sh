#!/bin/bash
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=4
#PBS -l walltime=8:00:00
#PBS -N PPO_Optuna_Optimization
#PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/ppo_optimization.log
#PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/ppo_optimization_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

# Run the Python script with PPO algorithm for Optuna optimization
python "/rds/general/user/lej23/home/fyp/Masters_RL/scripts/training/hyper_tune.py" --algo PPO --n_trials 1000 --timeout 28800 --n_jobs 16 --n_gpus 4

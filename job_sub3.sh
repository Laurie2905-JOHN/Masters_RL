# !/bin/bash
# PBS -l select=1:ncpus=32:mem=32gb:ngpus=0
# PBS -l walltime=24:00:00
# PBS -N A2C_Optuna_Optimization_TuneReward2
# PBS -o /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/a2c_optimization.log
# PBS -e /rds/general/user/lej23/home/fyp/Masters_RL/saved_models/hpc_output/a2c_optimization_error.log

cd $PBS_O_WORKDIR

module load tools/prod
module load anaconda3/personal

source activate MasterEnv

# Run the Python script with A2C algorithm for Optuna optimization
python "/rds/general/user/lej23/home/fyp/Masters_RL/scripts/training/hyper_tune.py" --algo A2C --n_trials 1000 --timeout 86300 --n_jobs 28

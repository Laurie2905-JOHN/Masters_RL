#!/bin/bash

# Define the training script path
TRAIN_SCRIPT="scripts/training/train.py"

# Define common training parameters
COMMON_ARGS="
  --algo="A2C" \
  --num_envs=8 \
  --total_timesteps=30000000 \
  --save_freq=10000 \
  --eval_freq=10000 "

# Top 1 trial hyperparameters
TRIAL_1_ARGS="
  --max_ingredients=5 \
  --action_update_factor=20 \
  --a2c_n_steps=5 \
  --gamma=0.9547360275466239 \
  --a2c_learning_rate=2.8218185699201902e-05 \
  --lr_schedule=constant \
  --a2c_ent_coef=0.030208648711824906 \
  --a2c_vf_coef=0.15426769757392855 \
  --a2c_max_grad_norm=0.5 \
  --a2c_gae_lambda=0.98 \
  --a2c_rms_prop_eps=0.0017596548252099434 \
  --a2c_use_rms_prop=False \
  --a2c_normalize_advantage=False \
  --a2c_ortho_init=True \
  --a2c_activation_fn=elu \
  --a2c_net_arch_width=512 \
  --a2c_net_arch_depth=2 \
  --best_prefix=trial_1 \
  --log_prefix=trial_1"

# Top 2 trial hyperparameters
TRIAL_2_ARGS="
  --max_ingredients=5 \
  --action_update_factor=20 \
  --a2c_n_steps=5 \
  --gamma=0.9788085126384909 \
  --a2c_learning_rate=2.6233912248483233e-05 \
  --lr_schedule=constant \
  --a2c_ent_coef=0.036300713827071895 \
  --a2c_vf_coef=0.1998798052828619 \
  --a2c_max_grad_norm=0.5 \
  --a2c_gae_lambda=0.8 \
  --a2c_rms_prop_eps=0.02044690735884977 \
  --a2c_use_rms_prop=False \
  --a2c_normalize_advantage=False \
  --a2c_ortho_init=True \
  --a2c_activation_fn=elu \
  --a2c_net_arch_width=512 \
  --a2c_net_arch_depth=3 \
  --best_prefix=trial_2 \
  --log_prefix=trial_2"

# Top 3 trial hyperparameters
TRIAL_3_ARGS="
  --max_ingredients=5 \
  --action_update_factor=20 \
  --a2c_n_steps=5 \
  --gamma=0.9754888085075119 \
  --a2c_learning_rate=4.467579173460465e-05 \
  --lr_schedule=constant \
  --a2c_ent_coef=0.09151807272195268 \
  --a2c_vf_coef=0.15722260514193426 \
  --a2c_max_grad_norm=0.5 \
  --a2c_gae_lambda=0.8 \
  --a2c_rms_prop_eps=0.07775725378306704 \
  --a2c_use_rms_prop=False \
  --a2c_normalize_advantage=False \
  --a2c_ortho_init=True \
  --a2c_activation_fn=elu \
  --a2c_net_arch_width=512 \
  --a2c_net_arch_depth=3 \
  --best_prefix=trial_3 \
  --log_prefix=trial_3"

# Run training for Top 1 trial
echo "Running training for Top 1 trial..."
python $TRAIN_SCRIPT $COMMON_ARGS $TRIAL_1_ARGS

# Run training for Top 2 trial
echo "Running training for Top 2 trial..."
python $TRAIN_SCRIPT $COMMON_ARGS $TRIAL_2_ARGS

# Run training for Top 3 trial
echo "Running training for Top 3 trial..."
python $TRAIN_SCRIPT $COMMON_ARGS $TRIAL_3_ARGS

echo "All training runs completed."

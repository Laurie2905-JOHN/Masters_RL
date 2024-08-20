#!/bin/bash

# Set up environment variables for the training parameters
export MAX_INGREDIENTS=7
export action_update_factor=20
export N_STEPS=16
export BATCH_SIZE=16
export GAMMA=0.995
export LEARNING_RATE=3.0818549847161666e-05
export ENT_COEF=1.2437243656681112e-06
export CLIP_RANGE=0.3
export N_EPOCHS=5
export GAE_LAMBDA=0.8
export MAX_GRAD_NORM=0.7
export VF_COEF=0.3783521604016542
export CLIP_RANGE_VF=0.3
export NORMALIZE_ADVANTAGE=True
export USE_SDE=True
export SDE_SAMPLE_FREQ=99
export TARGET_KL=0.08205318643418852
export POLICY_KWARGS_NET_ARCH_PI="16,16,16"
export POLICY_KWARGS_NET_ARCH_VF="16,16,16"
export POLICY_KWARGS_ACTIVATION_FN="relu"
export POLICY_KWARGS_ORTHO_INIT=True

# Call the Python training script with the necessary environment variables
python train.py --max_ingredients $MAX_INGREDIENTS \
                --action_update_factor $action_update_factor \
                --n_steps $N_STEPS \
                --batch_size $BATCH_SIZE \
                --gamma $GAMMA \
                --learning_rate $LEARNING_RATE \
                --ent_coef $ENT_COEF \
                --clip_range $CLIP_RANGE \
                --n_epochs $N_EPOCHS \
                --gae_lambda $GAE_LAMBDA \
                --max_grad_norm $MAX_GRAD_NORM \
                --vf_coef $VF_COEF \
                --clip_range_vf $CLIP_RANGE_VF \
                --normalize_advantage $NORMALIZE_ADVANTAGE \
                --use_sde $USE_SDE \
                --sde_sample_freq $SDE_SAMPLE_FREQ \
                --target_kl $TARGET_KL \
                --policy_kwargs_net_arch_pi $POLICY_KWARGS_NET_ARCH_PI \
                --policy_kwargs_net_arch_vf $POLICY_KWARGS_NET_ARCH_VF \
                --policy_kwargs_activation_fn $POLICY_KWARGS_ACTIVATION_FN \
                --policy_kwargs_ortho_init $POLICY_KWARGS_ORTHO_INIT

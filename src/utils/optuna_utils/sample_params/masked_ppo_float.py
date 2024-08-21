from typing import Dict, Any
import optuna
from torch import nn
from utils.train_utils import get_learning_rate

def sample_masked_ppo_params_float(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for Masked PPO hyperparams.

    :param trial:
    :return:
    """
    n_steps = trial.suggest_int("n_steps", 64, 512, step=64)
    batch_size = trial.suggest_int("batch_size", 256, 1024, step=256)
    gamma = 0.99
    learning_rate = trial.suggest_float("learning_rate", 4e-5, 0.001, log=True)
    lr_schedule_type = 'linear'
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-5, log=True)
    clip_range = trial.suggest_float("clip_range", 0.3, 0.4, step=0.01)
    n_epochs = trial.suggest_int("n_epochs", 10, 20, step=5)
    gae_lambda = 0.99
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1, step=0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 0.9, step=0.01)
    clip_range_vf = trial.suggest_float("clip_range_vf", 0.1, 0.3, step=0.01)
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    target_kl = trial.suggest_float("target_kl", 0.01, 0.03, step=0.005)

    # Orthogonal initialization
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    # Get learning rate schedule
    learning_rate = get_learning_rate(lr_schedule_type, learning_rate)

    # Define network architecture
    net_arch_width = trial.suggest_int("net_arch_width", 16, 256, step=16)
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 4)
    net_arch = dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "clip_range_vf": clip_range_vf,
        "normalize_advantage": normalize_advantage,
        "target_kl": target_kl,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

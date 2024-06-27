from typing import Dict, Any, Union, Callable

import optuna
from torch import nn

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.1, log=True)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    clip_range_vf = trial.suggest_categorical("clip_range_vf", [None, 0.1, 0.2, 0.3, 0.4])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    use_sde = trial.suggest_categorical("use_sde", [False, True])
    sde_sample_freq = trial.suggest_int("sde_sample_freq", -1, 100)
    target_kl = trial.suggest_float("target_kl", 0.01, 0.1, log=True)

    # Orthogonal initialization
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    # Ensure batch_size is not greater than n_steps
    if batch_size > n_steps:
        batch_size = n_steps

    # Adjust learning rate if schedule is linear
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Define network architecture
    net_arch_width = trial.suggest_categorical("net_arch_width", [8, 16, 32, 64, 128, 256, 512])
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
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "target_kl": target_kl,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

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

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical("n_steps", [5, 10, 20, 50, 100, 200])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7, 0.9, 1.0, 2, 5])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    rms_prop_eps = trial.suggest_float("rms_prop_eps", 1e-5, 1e-1, log=True)
    use_rms_prop = trial.suggest_categorical('use_rms_prop', [True, False])
    normalize_advantage = trial.suggest_categorical('normalize_advantage', [True, False])

    # Orthogonal initialization
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    # Adjust learning rate if schedule is linear
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Define network architecture
    net_arch_width = trial.suggest_categorical("net_arch_width", [8, 16, 32, 64, 128, 256, 512])
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    net_arch = dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": gae_lambda,
        "rms_prop_eps": rms_prop_eps,
        "use_rms_prop": use_rms_prop,
        "normalize_advantage": normalize_advantage,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

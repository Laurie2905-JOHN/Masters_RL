import argparse
# Importing from optuna study file
from optuna_study import optimize_ppo, optimize_a2c

def main(algo, n_trials, timeout, n_jobs, n_gpus):
    if algo == 'PPO':
        optimize_ppo(n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, n_gpus=n_gpus)
    elif algo == 'A2C':
        optimize_a2c(n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    else:
        raise ValueError("Unsupported algorithm specified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="A2C", help="Algorithm to optimize: PPO or A2C")
    parser.add_argument('--n_trials', type=int, default=1000, help="Number of trials for optimization")
    parser.add_argument('--timeout', type=int, default=259200, help="Timeout for optimization in seconds")
    parser.add_argument('--n_jobs', type=int, default=4, help="Number of parallel jobs for Optuna")
    parser.add_argument('--n_gpus', type=int, default=1, help="Number of GPUs to use")

    args = parser.parse_args()
    main(args.algo, args.n_trials, args.timeout, args.n_jobs, args.n_gpus)

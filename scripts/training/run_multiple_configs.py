import yaml
import copy
import os
import subprocess

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(config, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def run_experiment(hyperparams_file):
    # Assuming train.py can be run with a --hyperparams_file argument
    subprocess.run(['python', 'scripts/training/train.py', '--hyperparams_file', hyperparams_file])


def main():
    hyperparams_dir = "scripts/hyperparams"
    setup_file = os.path.join(hyperparams_dir, "setup.yaml")
    base_setup = load_yaml(setup_file)

    # Example variations
    env_names = ["SchoolMealSelection-v1", "SchoolMealSelection-v2"]
    algos = ['MASKED_PPO']
    reward_type = ['shaped']

    for env in env_names:
        for reward in reward_type:
            for algo in algos:
                if env == "SchoolMealSelection-v0" and algo == "MASKED_PPO":
                    continue
                modified_setup = copy.deepcopy(base_setup)
                modified_setup['env_name'] = env
                modified_setup['algo'] = algo
                modified_setup['reward_type'] = reward
                modified_setup['log_prefix'] = f"{env}_{reward}_{algo}"
                if env == "SchoolMealSelection-v2":
                    modified_setup['max_episode_steps'] = 1000

                # Create a unique setup file for this run
                temp_setup_path = f'temp_setup_{env}_{reward}_{algo}.yaml'
                save_yaml(modified_setup, temp_setup_path)

                # Run the experiment with the modified setup
                run_experiment(temp_setup_path)

                # Remove the temporary setup file if needed
                os.remove(temp_setup_path)

if __name__ == '__main__':
    main()

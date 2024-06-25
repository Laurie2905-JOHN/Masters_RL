import optuna
import pickle
import matplotlib.pyplot as plt
import os
from utils.train_utils import get_unique_directory

study_name = "PPO_study1"  # Replace with your actual study name

# Define the path to load the study
study_file_path = f"saved_models/optuna/{study_name}/study.pkl"

# Load the study using pickle
with open(study_file_path, 'rb') as f:
    study = pickle.load(f)

# Retrieve the best trial
best_trial = study.best_trial

# Print the best parameters
print("Best trial:")
print(f"  Trial number: {best_trial.number}")
print(f"  Value: {best_trial.value}")

print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Retrieve all completed trials
trials = study.get_trials(deepcopy=False)
completed_trials = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]

# Determine if the study is minimizing or maximizing
direction = study.direction

# Sort trials by objective value (reward)
if direction == optuna.study.StudyDirection.MINIMIZE:
    sorted_trials = sorted(completed_trials, key=lambda x: x.value)
else:
    sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)

# Select the top 10 trials
top_trials = sorted_trials[:10]

# Print the top 10 trials
for i, trial in enumerate(top_trials):
    print(f"\nTop {i+1} trial:")
    print(f"  Trial number: {trial.number}")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# Extract all hyperparameter names
all_params = set()
for trial in top_trials:
    all_params.update(trial.params.keys())

# Determine the number of subplots (grid size)
num_params = len(all_params)
cols = 6
rows = (num_params + cols - 1) // cols

# Create a tiled plot for each hyperparameter
fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
axes = axes.flatten()

# Colors for the top 3 trials
colors = ['green', 'red', 'blue']
handles = []

for i, param in enumerate(all_params):
    param_values = []
    rewards = []
    
    for trial in top_trials:
        if param in trial.params:
            param_values.append(trial.params[param])
            rewards.append(trial.value)
    
    ax = axes[i]
    ax.scatter(param_values, rewards, label='Other trials')

    # Highlight the top 3 trials
    for rank in range(3):
        if param in top_trials[rank].params:
            handle = ax.scatter(top_trials[rank].params[param], top_trials[rank].value, color=colors[rank], s=100, label=f'Trial {rank + 1}')
            if i == 0:
                handles.append(handle)
    
    ax.set_title(f'{param}')
    ax.set_xlabel(param)
    ax.set_ylabel('Reward')
    ax.grid(True)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()

# Add a single legend for the whole figure
fig.legend(handles=handles, labels=[f'Best Trial {rank + 1}' for rank in range(3)], loc='lower center', ncol=3)

# Save the figure
base, filename = get_unique_directory(os.path.dirname(study_file_path), "hyperparam_rewards_trial1", ".png")
output_path = os.path.join(base, filename)
plt.savefig(output_path)
plt.show()

print(f"Figure saved to {output_path}")

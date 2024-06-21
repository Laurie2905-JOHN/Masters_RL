import pickle
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import os

# Define the study path
study_path = "saved_models/optuna/A2C_parralel"

# Ensure the directory exists
os.makedirs(study_path, exist_ok=True)

# Load the study
with open(f"{study_path}/study.pkl", "rb") as f:
    study = pickle.load(f)

# Generate the figures
fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)

# Save figures to files in the specified directory
fig1.write_image(f"{study_path}/optimization_history.png")
fig2.write_image(f"{study_path}/param_importances.png")
fig3.write_image(f"{study_path}/parallel_coordinate.png")

import os
import pickle
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_timeline

# Define the study path
study_path = "notebooks"

# Ensure the directory exists
os.makedirs(study_path, exist_ok=True)

# Load the study
study_filename = os.path.join(study_path, "4500_study.pkl")
with open(study_filename, "rb") as f:
    study = pickle.load(f)

# Generate the figures
fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)
fig4 = plot_timeline(study)

# Save figures to files in the specified directory
fig1.write_image(os.path.join(study_path, "optimization_history.png"))
fig2.write_image(os.path.join(study_path, "param_importances.png"))
fig3.write_image(os.path.join(study_path, "parallel_coordinate.png"))
fig4.write_image(os.path.join(study_path, "timeline.png"))

print(f"Figures saved to {study_path}")

# import pickle
# from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_timeline
# import os

# # Define the study path
# study_path = "saved_models/optuna/PPO_study1"

# # Ensure the directory exists
# os.makedirs(study_path, exist_ok=True)

# # Load the study
# with open(f"{study_path}/study.pkl", "rb") as f:
#     study = pickle.load(f)

# # Generate the figures
# fig1 = plot_optimization_history(study)
# fig2 = plot_param_importances(study)
# fig3 = plot_parallel_coordinate(study)
# fig4 = plot_timeline(study)

# # Save figures to files in the specified directory
# fig1.write_image(f"{study_path}/optimization_history.png")
# fig2.write_image(f"{study_path}/param_importances.png")
# fig3.write_image(f"{study_path}/parallel_coordinate.png")
# fig4.write_image(f"{study_path}/timeline.png")


import optuna
import os
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, plot_terminator_improvement, plot_timeline

# Load the study
study_name = "A2C_parralel1"  # replace with your study name

# Define the directory containing the journal log
db_dir = f"saved_models/optuna/{study_name}/journal_storage"
journal_file = os.path.join(db_dir, "journal.log")

# Set up the JournalStorage
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(journal_file)
)


study = optuna.load_study(study_name=study_name, storage=storage)

# Generate the figures
fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)
# fig4 = plot_terminator_improvement(study, plot_error=True)
fig5 = plot_timeline(study)
# fig6 = plot_rank(study, params=["x", "y"])
# Define the path to save the figures
study_path = f"saved_models/optuna/{study_name}"

# Ensure the directory exists
os.makedirs(study_path, exist_ok=True)

# Save figures to files in the specified directory
fig1.write_image(f"{study_path}/optimization_history.png")
fig2.write_image(f"{study_path}/param_importances.png")
fig3.write_image(f"{study_path}/parallel_coordinate.png")
# fig4.write_image(f"{study_path}/terminator_improvement.png")
fig5.write_image(f"{study_path}/timeline.png")
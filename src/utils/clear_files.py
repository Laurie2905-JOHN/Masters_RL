import os
import shutil
import argparse

def clear_directory(directory_list):
    # Warning message
    print(f"WARNING: This will permanently delete all files and subdirectories in: {directory_list}")
    confirmation = input("Are you sure you want to continue? (yes/no): ")
    # Proceed based on user input
    if confirmation.lower() == 'yes':
        for directory in directory_list:
            directory_path = os.path.abspath(directory)
            # Check if the directory exists
            if not os.path.exists(directory_path):
                print(f"The directory {directory_path} does not exisyest.")
                continue
            
            # List all files and directories in the specified path
            items = os.listdir(directory_path)
            if not items:
                print(f"The directory {directory_path} is already empty.")
                continue
    
            for item in items:
                item_path = os.path.join(directory_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"All contents of {directory_path} have been deleted.")
        else:
            print("Operation cancelled.")

if __name__ == "__main__":
    
    # directory_list = ["saved_models/tensorboard", "saved_models/best_models", "saved_models/checkpoints", "saved_models/hpc_output", "saved_models/reward"]
    directory_list = ["saved_models/tensorboard", "saved_models/reward", "saved_models/best_models", "saved_models/checkpoints"]
    clear_directory(directory_list)


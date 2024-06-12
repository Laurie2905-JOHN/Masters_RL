import subprocess
import time
import pandas as pd
import psutil

# List of different number of environments to test
num_envs_list = []
for i in range(1, 50):
    num_envs_list.append(i)
    
num_repeats = 10

# Results storage
results = []

# Base arguments for train.py
base_args = [
    "python", "scripts/training/train.py",
    "--env_name", "SchoolMealSelection-v1",
    "--algo", "A2C",
    "--total_timesteps", "150000",
    "--save_freq", "1000",
    "--eval_freq", "1000",
    "--seed", "1",
    "--max_episode_steps", "1000",
    "--memory_monitor", "True",
    "--reward_save_interval", "2500"
    "--save_prefix", "test_" + str(time.time())
]

# Function to measure CPU utilization
def measure_cpu_utilization(process, duration):
    cpu_usage = []
    start_time = time.time()
    while time.time() - start_time < duration:
        if process.poll() is not None:
            break
        cpu_usage.append(psutil.cpu_percent(interval=1))
    return sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

# Run training for each num_envs value
for num_envs in num_envs_list:
    total_training_time = 0
    total_cpu_utilization = 0
    for _ in range(num_repeats):
        args = base_args + ["--num_envs", str(num_envs)]
        
        start_time = time.time()
        process = subprocess.Popen(args)
        
        # Measure CPU utilization during the process execution
        avg_cpu_utilization = measure_cpu_utilization(process, duration=15)  # Adjust duration as needed
        
        process.wait()  # Ensure the process has completed
        end_time = time.time()
        
        training_time = end_time - start_time
        total_training_time += training_time
        total_cpu_utilization += avg_cpu_utilization
    
    avg_training_time = total_training_time / num_repeats
    avg_cpu_utilization = total_cpu_utilization / num_repeats
    
    results.append({
        "num_envs": num_envs,
        "avg_training_time": avg_training_time,
        "avg_cpu_utilization": avg_cpu_utilization,
        "return_code": process.returncode
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("training_speed_results.csv", index=False)

# Print results
print(results_df)

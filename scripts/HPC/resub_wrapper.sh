#!/bin/bash

# Job submission script name
JOB_SCRIPT="job_sub.sh"
# Wall time in seconds (3 hours = 10800 seconds)
WALL_TIME=10800

# Function to check the job status
check_job_status() {
  local job_info=$(qstat -f $1 2>&1)
  
  if echo "$job_info" | grep -q 'job_state = C'; then
    if echo "$job_info" | grep -q 'exit_status = 0'; then
      echo "Job completed successfully."
      return 0
    else
      echo "Job did not complete successfully."
      return 1
    fi
  else
    return 2
  fi
}

# Submit the job initially
JOB_ID=$(qsub $JOB_SCRIPT)
echo "Submitted job with ID: $JOB_ID"

# Start time of the job
START_TIME=$(date +%s)

# Loop to resubmit the job if necessary
while true; do
  sleep 300  # Wait for 5 minutes before checking the job status

  # Check if the wall time has been exceeded
  CURRENT_TIME=$(date +%s)
  ELAPSED_TIME=$(($CURRENT_TIME - $START_TIME))

  if [ $ELAPSED_TIME -ge $WALL_TIME ]; then
    echo "Wall time exceeded. Resubmitting job..."
    JOB_ID=$(qsub $JOB_SCRIPT)
    echo "Resubmitted job with ID: $JOB_ID"
    START_TIME=$(date +%s)
  else
    check_job_status $JOB_ID
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
      break
    elif [ $STATUS -eq 1 ]; then
      echo "Job did not complete successfully. Resubmitting job..."
      JOB_ID=$(qsub $JOB_SCRIPT)
      echo "Resubmitted job with ID: $JOB_ID"
      START_TIME=$(date +%s)
    else
      echo "Job is still running."
    fi
  fi
done

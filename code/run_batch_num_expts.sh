#!/bin/bash

# Array to store all process IDs
pids=()

# Run the Python script for each directory in parallel
for i in {4..17}; do
  # Format number with leading zeros
  dirnum=$(printf "%03d" $i)
  
  # Run the Python command in the background
  python3 core_micropk.py "../num_expt/cpBL$dirnum" &
  
  # Store the process ID
  pids+=($!)
  
  echo "Started task for cpBL$dirnum with PID ${pids[-1]}"
done

# Wait for all background processes to complete
echo "Waiting for all tasks to complete..."
for pid in "${pids[@]}"; do
  wait $pid
  echo "Process $pid completed"
done

echo "All tasks completed successfully"

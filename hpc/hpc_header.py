# generate a CSV file with tasks for the Slurm array job
import h5py
import pandas as pd
import os
import numpy as np
import torch

def select_groups_and_generate_tasks_reduced(h5_path, csv_output_path):
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    group_stats = []
    with h5py.File(h5_path, 'r') as h5f:
        # 1. Inspect the file and gather statistics
        for group_id in h5f.keys():
            if isinstance(h5f[group_id], h5py.Group):
                neuron_count = len(h5f[group_id].keys())
                group_stats.append({"group_id": group_id, "neuron_count": neuron_count})

    # 2. Display groups to the user
    stats_df = pd.DataFrame(group_stats)
    print("\n--- Available Groups in HDF5 ---")
    print(stats_df.to_string(index=False))
    print("--------------------------------")

    # 3. Ask for input
    user_input = input("\nEnter the group_ids to include (comma-separated), or type 'ALL': ").strip()

    if user_input.upper() == 'ALL':
        selected_groups = stats_df['group_id'].tolist()
    else:
        selected_groups = [g.strip() for g in user_input.split(',')]

    # 4. Filter and Generate Tasks
    tasks = []
    task_counter = 0
    
    with h5py.File(h5_path, 'r') as h5f:
        for group_id in selected_groups:
            if group_id in h5f:
                for neuron_id_str in h5f[group_id].keys():
                    tasks.append({
                        "task_id": task_counter,
                        "group_id": group_id,
                        "neuron_id": int(neuron_id_str)
                    })
                    task_counter += 1
            else:
                print(f"Warning: Group '{group_id}' not found in HDF5. Skipping.")

    if tasks:
        df = pd.DataFrame(tasks)
        df = df[["task_id", "group_id", "neuron_id"]]
        df.to_csv(csv_output_path, index=False)
        print(f"\nSuccess: {len(df)} tasks saved to {csv_output_path} for groups: {selected_groups}")
    else:
        print("No tasks generated.")


def get_task_params_reduced(csv_path, task_id):
    """
    Reads the tasks.csv file and returns the group_id and neuron_id 
    for a specific task_id.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Filter for the specific task_id
        task_row = df[df['task_id'] == task_id]
        
        if task_row.empty:
            print(f"Error: Task ID {task_id} not found in {csv_path}")
            return None, None
            
        # Extract values
        group_id = task_row.iloc[0]['group_id']
        neuron_id = str(task_row.iloc[0]['neuron_id'])
        
        return group_id, neuron_id

    except Exception as e:
        print(f"An error occurred while reading the task file: {e}")
        return None, None
    

def read_event_times_reduced(h5_path, group_id, neuron_id):
    """
    Reads the event times for a specific neuron from an HDF5 file.
    
    Args:
        h5_path (str): Path to the .h5 file.
        group_id (str): The name of the group (e.g., 'BC').
        neuron_id (str): The unique identifier for the neuron.
        
    Returns:
        numpy.ndarray: Array of event times, or None if the path is not found.
    """
    try:
        with h5py.File(h5_path, 'r') as h5f:
            # Construct the internal HDF5 path
            # HDF5 keys are stored as strings
            dataset_path = f"{group_id}/{neuron_id}"
            
            if dataset_path in h5f:
                # Slicing with [:] loads the dataset into memory as a numpy array
                return h5f[dataset_path][:]
            else:
                print(f"Dataset {dataset_path} not found in {h5_path}")
                return None
                
    except Exception as e:
        print(f"Failed to read HDF5 file: {e}")
        return None
    
if __name__ == "__main__":

    # Before running slurm script, generate the tasks CSV file by selecting groups from the HDF5 file
    select_groups_and_generate_tasks_reduced('data/event_times_data.h5', 'hpc/tasks_reduced.csv')

    # Example of how to read task parameters in the Slurm job script
    task_id = 5  # This would typically come from the Slurm environment variable, e.g., os.environ['SLURM_ARRAY_TASK_ID']
    group_id, neuron_id = get_task_params_reduced('hpc/tasks_reduced.csv', task_id)
    if group_id is not None:
        print(f"Lookup task {task_id}: group_id: {group_id}, neuron_id: {neuron_id}")

    event_times = read_event_times_reduced('data/event_times_data.h5', group_id, neuron_id)
    print(f"Event times for {group_id}/{neuron_id}: {event_times[:10]}...")  # Print first 10 event times for verification


import argparse
from copy import deepcopy
import json
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from src.te_tpp import Ln_estimation_yy
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt
from hpc.hpc_header import get_task_params_reduced, read_event_times_reduced

def load_best_config_reduced(file_path):
    """
    Loads the optimization configuration dictionary from a JSON-formatted text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No configuration file found at: {file_path}")
        
    with open(file_path, "r") as f:
        config = json.load(f)
    
    return config

def run_multiple_estimation_reduced(target_events, configs, task_id, n_runs=10, seed=42):
    """
    Runs Ln_yy estimation multiple times. 
    Can resume unfinished runs by checking the existing CSV file.
    Saves per-run results incrementally to prevent data loss.
    """

    os.makedirs("results/hpc_runs-reduced", exist_ok=True)
    output_file = f"results/hpc_runs-reduced/runs_reduced_{task_id}.csv"

    start_run = 0
    run_results = []

    # Check for existing progress to resume
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty and "run" in existing_df.columns:
                start_run = int(existing_df["run"].max())
                run_results = existing_df.to_dict('records')
                print(f"Found existing results. Resuming from run {start_run + 1}...")
        except pd.errors.EmptyDataError:
            print("Existing file is empty. Starting from run 1...")

    if start_run >= n_runs:
        print(f"All {n_runs} runs have already been completed for task {task_id}. Exiting.")
        return pd.DataFrame(run_results)

    print(f"--- Starting Multiple Runs Estimation (Runs {start_run + 1} to {n_runs}) ---")

    for run in tqdm(range(start_run, n_runs)):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        run_start_time = time.time()
        run_seed = seed + (run+1) * 1000
        
        ln_yy_sec, log_loss_yy = Ln_estimation_yy(
            event_time=[target_events],  # Only target events are needed for the reduced model
            configs=deepcopy(configs),
            seed=run_seed,
        )
        
        run_duration = time.time() - run_start_time
        
        print(f"Run {run+1} : ln_yy = {ln_yy_sec:.3f}")
        print(f"Loss - yy: {log_loss_yy:.3f}")
        print(f"Run {run+1} completed in {run_duration/60:.2f} minutes.")

        # Create a dictionary for the current run
        current_result = {
            "run": run + 1,
            "ln_yy_sec": ln_yy_sec,
            "loss_yy": log_loss_yy, 
            "run_duration_sec": run_duration,
        }
        run_results.append(current_result)

        # Save incrementally 
        write_header = not os.path.exists(output_file)
        pd.DataFrame([current_result]).to_csv(output_file, mode='a', header=write_header, index=False)

    # Convert the combined results to DataFrame for final reporting
    results_df = pd.DataFrame(run_results)

    # --- REPORTING ---
    print("\n--- Multiple Runs Summary ---")
    print(results_df[["ln_yy_sec", "loss_yy"]].describe().T)
    
    print(f"\nResults saved to {output_file}")
    return results_df


# User needs to make sure same parameters (except --num_runs) are used for the multiple runs as the runs continues after interruption. The best way is to use the same command line arguments for the multiple runs as the optimization.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default="data/event_times_data.h5", help="Path to the HDF5 file containing event times")
    parser.add_argument("--task_id", type=int, required=True, help="Slurm Array Task ID")
    parser.add_argument("--history_length", type=int, default=None, help="History length for TE estimation")
    parser.add_argument("--data_time_length", type=int, default=None, help="Total time of the sequences in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs("results/hpc_runs-reduced", exist_ok=True)

    # Read task parameters from task file
    group_id, neuron_id = get_task_params_reduced('hpc/tasks_reduced.csv', args.task_id)
    if group_id is not None:
        print(f"Multiple runs task {args.task_id}: group_id: {group_id}, neuron_id: {neuron_id}")
    else:
        print(f"Warning: Task ID {args.task_id} not found in task file. Exiting.")
        exit(0) 
    
    # Load event times for the specified group_id and neuron_id
    target_events = read_event_times_reduced(args.data_file_path, group_id, neuron_id)
    target_events = torch.tensor(target_events, dtype=torch.float)

    # Print summary statistics
    print("\n--- Data Summary ---")
    print(f"Total events for target process {len(target_events)}")
    print(f"Data Time: {args.data_time_length} seconds")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the best configuration for the reduced model using dynamic task_id
    config_path = f"results/opt-reduced/opt_reduced_{args.task_id}_best_config.txt"
    best_configs = load_best_config_reduced(config_path)
    
    # Ensure the model runs on the currently available device
    best_configs["device"] = device 

    # Handle Conditional Overwrites
    # Only overwrite if the user specified them in command line (args is not None)
    if args.history_length is not None:
        print(f"Overwriting history_length: {args.history_length}")
        best_configs["history_length"] = args.history_length
    
    if args.data_time_length is not None:
        print(f"Overwriting data_time_length: {args.data_time_length}")
        if "data_prep_config" in best_configs:
            best_configs["data_prep_config"]["total_time"] = args.data_time_length

    # Run multiple estimations and save results
    run_multiple_estimation_reduced(
        target_events, 
        best_configs, 
        task_id=args.task_id, 
        n_runs=args.num_runs, 
        seed=args.seed
    )
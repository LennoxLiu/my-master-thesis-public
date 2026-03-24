import argparse
from copy import deepcopy
import json
import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from src.te_tpp import Ln_estimation_yyx
import numpy as np
from hpc.hpc_header import get_task_params_full, read_event_times_full

def load_best_config_full(file_path):
    """
    Loads the optimization configuration dictionary from a JSON-formatted text file.
    """
    if not os.path.exists(file_path):
        # Instead of raising an error that crashes Slurm, 
        # print a message and exit cleanly.
        print(f"Skipping: No configuration file found at {file_path}")
        sys.exit(0)
        
    with open(file_path, "r") as f:
        config = json.load(f)
    
    return config

def run_multiple_estimation_full(source_events, target_events, configs, task_id, n_runs=10, seed=42, surrogate=False):
    """
    Runs Ln_yyx estimation multiple times. 
    Can resume unfinished runs by checking the existing CSV file.
    Saves per-run results incrementally to prevent data loss.
    """

    if surrogate:
        print("Running with surrogate data. Results will reflect shuffled event times.")
        os.makedirs("results/runs-full-surrogate", exist_ok=True)
        output_file = f"results/runs-full-surrogate/runs_full_surrogate_{task_id}.csv"
    else:
        os.makedirs("results/runs-full", exist_ok=True)
        output_file = f"results/runs-full/runs_full_{task_id}.csv"

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
        
        ln_yyx_sec, log_loss_yyx = Ln_estimation_yyx(
            event_time=[target_events, source_events],  # Source and target events needed for full model
            configs=deepcopy(configs),
            seed=run_seed,
        )
        
        run_duration = time.time() - run_start_time
        
        print(f"Run {run+1} : ln_yyx = {ln_yyx_sec:.3f}")
        print(f"Loss - yyx: {log_loss_yyx:.3f}")
        print(f"Run {run+1} completed in {run_duration/60:.2f} minutes.")

        # Create a dictionary for the current run
        current_result = {
            "run": run + 1,
            "ln_yyx_sec": ln_yyx_sec,
            "loss_yyx": log_loss_yyx, 
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
    print(results_df[["ln_yyx_sec", "loss_yyx"]].describe().T)
    
    print(f"\nResults saved to {output_file}")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default="data/event_times_data.h5", help="Path to the HDF5 file containing event times")
    parser.add_argument("--task_id", type=int, required=True, help="Slurm Array Task ID")
    parser.add_argument("--history_length", type=int, default=None, help="History length for TE estimation")
    parser.add_argument("--data_time_length", type=int, default=None, help="Total time of the sequences in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs to perform")
    parser.add_argument("--surrogate", action="store_true", help="Enable surrogate data for estimation")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Read task parameters from task file
    s_group, s_neuron, t_group, t_neuron = get_task_params_full('hpc/tasks_full.csv', args.task_id)
    if s_group is not None:
        print(f"Multiple runs task {args.task_id}: source_group: {s_group}, source_neuron: {s_neuron}, target_group: {t_group}, target_neuron: {t_neuron}")
    else:
        print(f"Warning: Task ID {args.task_id} not found in task file. Exiting.")
        exit(0) 
    
    # Load event times for the specified groups and neurons
    source_events, target_events = read_event_times_full(args.data_file_path, s_group, s_neuron, t_group, t_neuron)
    source_events = torch.tensor(source_events, dtype=torch.float)
    target_events = torch.tensor(target_events, dtype=torch.float)

    # Print summary statistics
    print("\n--- Data Summary ---")
    print(f"Total events for source process {len(source_events)}")
    print(f"Total events for target process {len(target_events)}")
    print(f"Data Time: {args.data_time_length} seconds")

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load the best configuration for the full model using dynamic task_id
    config_path = f"results/opt-full/opt_full_{args.task_id}_best_config.txt"
    best_configs = load_best_config_full(config_path)
    
    # Ensure the model runs on the currently available device
    best_configs["device"] = device 

    # Handle Conditional Overwrites
    if args.history_length is not None:
        print(f"Overwriting history_length: {args.history_length}")
        best_configs["history_length"] = args.history_length
    
    if args.data_time_length is not None:
        print(f"Overwriting data_time_length: {args.data_time_length}")
        if "data_prep_config" in best_configs:
            best_configs["data_prep_config"]["total_time"] = args.data_time_length

    if args.surrogate:
        print("Using surrogate data for estimation.")
        best_configs["data_prep_config"]["shuffle"] = True

    # Run multiple estimations and save results
    run_multiple_estimation_full(
        source_events,
        target_events, 
        best_configs, 
        task_id=args.task_id, 
        n_runs=args.num_runs, 
        seed=args.seed,
        surrogate=args.surrogate
    )
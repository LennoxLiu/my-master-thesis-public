
from copy import deepcopy
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from src.te_tpp import Ln_estimation_yy
from scipy.stats import rankdata
import numpy as np
import matplotlib.pyplot as plt

def load_best_config_reduced(file_path):
    """
    Loads the optimization configuration dictionary from a JSON-formatted text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No configuration file found at: {file_path}")
        
    with open(file_path, "r") as f:
        config = json.load(f)
    
    return config


# Example usage:
# task_id = 2  # Replace with your actual task ID
# config_path = f"results/opt-reduced/opt_reduced_{task_id}_best_config.txt"
# best_configs = load_best_config_reduced(config_path)

# # Accessing values:
# print(best_configs)

def run_multiple_estimation_reduced(target_events, configs, n_runs=10, seed=42):
    """
    Runs Ln_yy estimation multiple times with rank-sum weighted averaging based on validation loss.
    Saves both per-run results and weighted summary statistics to CSV.
    """

    os.makedirs("results/multi_runs-reduced", exist_ok=True)

    print(f"--- Starting {n_runs} Multiple Runs Estimation ---")
    run_results = []

    for run in tqdm(range(n_runs)):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        run_start_time = time.time()
        run_seed = seed + (run+1) * 1000
        
        ln_yy_sec, log_loss_yy = Ln_estimation_yy(
            event_time=[target_events],  # Only target events are needed for the reduced model
            configs=deepcopy(configs),
            seed=seed,
        )
        
        run_duration = time.time() - run_start_time
        
        print(f"Run {run+1} : ln_yy = {ln_yy_sec:.3f}")
        print(f"Loss - yy: {log_loss_yy:.3f}")
        print(f"Run {run+1} completed in {run_duration/60:.2f} minutes.")

        run_results.append({
            "run": run + 1,
            "ln_yy_sec": ln_yy_sec,
            "loss_yy": log_loss_yy, 
            "run_duration_sec": run_duration,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(run_results)

    # # --- WEIGHTED CALCULATION (RANK-SUM) ---
    # def calculate_weights(losses):
    #     # Rank negative losses so the lowest loss gets the highest rank
    #     ranks = rankdata(-np.array(losses))
    #     return ranks / np.sum(ranks)

    # # Calculate weights based on specific losses
    # weights_yy = calculate_weights(results_df['loss_yy'].values)

    # # 1. Weighted Means
    # w_h_yy = np.sum(weights_yy * results_df['ln_yy_sec'])
    
    # # 2. Weighted Standard Deviation (Unbiased Reliability Weighting)
    # def get_weighted_std(values, weights, w_mean):
    #     variance = np.sum(weights * (values - w_mean)**2)
    #     # Bessel's correction for weighted data
    #     correction = 1 / (1 - np.sum(weights**2))
    #     return np.sqrt(variance * correction)

    # std_h_yy = get_weighted_std(results_df['ln_yy_sec'].values, weights_yy, w_h_yy)
    
    # # --- SAVE WEIGHTS AND STATS TO DATAFRAME ---
    # # Add weights as columns so you can see which run contributed most
    # results_df['weight_yy'] = weights_yy

    # # Append a Summary Row at the bottom for easy CSV reading
    # ln_mean = results_df['ln_yy_sec'].mean()
    # mean_data = {
    #     "run": "SIMPLE_MEAN",
    #     "ln_yy_sec": ln_mean,
    #     "loss_yy": results_df['loss_yy'].mean(),
    # }

    # ln_std = results_df['ln_yy_sec'].std()
    # std_data = {
    #     "run": "SIMPLE_STD",
    #     "ln_yy_sec": results_df['ln_yy_sec'].std(),
    #     "loss_yy": results_df['loss_yy'].std(),
    # }

    # weighted_mean = {
    #     "run": "WEIGHTED_MEAN",
    #     "ln_yy_sec": w_h_yy,
    #     "loss_yy": results_df['loss_yy'].mean(), # Simple mean for reference
    #     "run_duration_sec": results_df['run_duration_sec'].sum()
    # }
    
    # # Also add a row for the Weighted STD
    # weighted_std = {
    #     "run": "WEIGHTED_STD",
    #     "ln_yy_sec": std_h_yy,
    # }
    
    # Final CSV Output: Per-run results followed by the Weighted Summary
    # final_output_df = pd.concat([results_df, pd.DataFrame([mean_data, std_data,weighted_mean , weighted_std])], ignore_index=True)
    
    results_df.to_csv("results/multi_runs-reduced/multiple_runs_results_reduced.csv", index=False)

    # --- REPORTING ---
    print("\n--- Multiple Runs Summary ---")
    print(results_df[["ln_yy_sec", "loss_yy"]].describe().T)
    
    
    # # --- VISUALIZATION ---
    # fig, ax = plt.subplots(figsize=(10, 6))
    # results_df[["ln_yy_sec"]].plot(kind='box', ax=ax)
    # ax.axhline(ln_mean, color='r', linestyle='--', label=f'Simple Mean ({ln_mean:.4f})')
    
    # # Fill standard deviation across the ENTIRE x-axis
    # ax.axhspan(ln_std - ln_std, ln_std + ln_std, 
    #            color='red', alpha=0.15, label='Simple Std Dev', zorder=1)
    
    # ax.set_title("Transfer Entropy - Multiple Runs")
    # ax.set_ylabel("Ln_yy (nats/sec)")
    # ax.legend()
    # fig.savefig("results/Ln_yy_weighted_boxplot.png")
    # plt.close(fig)

    print(f"\nResults saved to results/multiple_runs_results.csv")
    return results_df


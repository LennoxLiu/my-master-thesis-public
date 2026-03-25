import os
import pandas as pd
import numpy as np
import h5py
from scipy import stats

# Configuration paths
TASKS_FULL_PATH = "hpc/tasks_full.csv"
INPUT_H5_PATH = "results/multi_runs_results.h5"
OUTPUT_CSV_PATH = "results/te_results_hpc.csv"

def get_bootstrap_ci(data, n_iterations=2000, ci=0.95):
    """Calculates the bootstrap confidence interval for the mean."""
    if len(data) < 2:
        return 0.0, 0.0
    
    boot_means = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    
    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100
    
    lower = np.percentile(boot_means, lower_percentile)
    upper = np.percentile(boot_means, upper_percentile)
    return lower, upper

def linear_to_db(snr_linear):
    """
    Converts linear SNR to dB using the amplitude formula: 20 * log10(SNR).
    Returns -np.inf for SNR <= 0 to represent a lack of detectable signal.
    """
    if snr_linear <= 0:
        return -np.inf
    return 20 * np.log10(snr_linear)

def main():
    if not os.path.exists(TASKS_FULL_PATH) or not os.path.exists(INPUT_H5_PATH):
        print("Required input files are missing.")
        return

    tasks_full = pd.read_csv(TASKS_FULL_PATH)
    results = []

    with h5py.File(INPUT_H5_PATH, 'r') as h5f:
        for _, row in tasks_full.iterrows():
            task_id = int(row['task_id'])
            src_group, src_neuron = row['source_group_id'], int(row['source_neuron_id'])
            tgt_group, tgt_neuron = row['target_group_id'], int(row['target_neuron_id'])

            full_grp_name = f"full/{src_group}_{src_neuron}_to_{tgt_group}_{tgt_neuron}"
            reduced_grp_name = f"full_surrogate/{src_group}_{src_neuron}_to_{tgt_group}_{tgt_neuron}"

            if full_grp_name in h5f and reduced_grp_name in h5f:
                try:
                    ln_yyx_sec = h5f[full_grp_name]['ln_yyx_sec'][:]
                    ln_yyx_surrogate_sec = h5f[reduced_grp_name]['ln_yyx_sec'][:]
                    run_duration = h5f[full_grp_name]['run_duration_sec'][:]
                    
                    min_len = min(len(ln_yyx_sec), len(ln_yyx_surrogate_sec))
                    te_array = ln_yyx_sec[:min_len] - ln_yyx_surrogate_sec[:min_len]

                    # Basic Metrics
                    te_mean = np.mean(te_array)
                    te_std = np.std(te_array, ddof=1) if min_len > 1 else 0.0
                    
                    # Uncertainty Metrics
                    te_sem = stats.sem(te_array) if min_len > 1 else 0.0
                    te_iqr = stats.iqr(te_array) if min_len > 0 else 0.0
                    ci_low, ci_high = get_bootstrap_ci(te_array)

                    # SNR and dB calculation
                    # Note: SNR can be negative if te_mean is negative
                    te_snr_linear = te_mean / te_std if te_std > 0 else 0.0
                    te_snr_db = linear_to_db(te_snr_linear)

                    results.append({
                        "task_id": task_id,
                        "source_group_id": src_group,
                        "source_neuron_id": src_neuron,
                        "target_group_id": tgt_group,
                        "target_neuron_id": tgt_neuron,
                        "te_mean": te_mean,
                        "te_std": te_std,
                        "te_sem": te_sem,
                        "te_iqr": te_iqr,
                        "te_ci_lower": ci_low,
                        "te_ci_upper": ci_high,
                        "te_snr_linear": te_snr_linear,
                        "te_snr_db": te_snr_db,
                        "mean_runtime": np.mean(run_duration)
                    })
                except KeyError as e:
                    print(f"Missing dataset {e} for task {task_id}.")
            else:
                print(f"Missing group for task {task_id}.")

    if results:
        results_df = pd.DataFrame(results)
        columns_order = [
            "task_id", "source_group_id", "source_neuron_id", 
            "target_group_id", "target_neuron_id", 
            "te_mean", "te_std", "te_sem", "te_iqr", 
            "te_ci_lower", "te_ci_upper", 
            "te_snr_linear", "te_snr_db", "mean_runtime"
        ]
        results_df[columns_order].to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
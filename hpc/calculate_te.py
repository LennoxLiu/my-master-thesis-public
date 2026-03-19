import os
import pandas as pd
import numpy as np
import h5py

# Configuration paths
TASKS_FULL_PATH = "hpc/tasks_full.csv"
INPUT_H5_PATH = "results/multi_runs_results.h5"
OUTPUT_CSV_PATH = "results/te_results_hpc.csv"

def main():
    if not os.path.exists(TASKS_FULL_PATH) or not os.path.exists(INPUT_H5_PATH):
        print("Required input files are missing. Please ensure tasks_full.csv and the h5 file exist.")
        return

    tasks_full = pd.read_csv(TASKS_FULL_PATH)
    results = []

    with h5py.File(INPUT_H5_PATH, 'r') as h5f:
        for _, row in tasks_full.iterrows():
            task_id = int(row['task_id'])
            src_group = row['source_group_id']
            src_neuron = int(row['source_neuron_id'])
            tgt_group = row['target_group_id']
            tgt_neuron = int(row['target_neuron_id'])

            # H5 paths
            full_grp_name = f"full/{src_group}_{src_neuron}_to_{tgt_group}_{tgt_neuron}"
            reduced_grp_name = f"reduced/{tgt_group}_{tgt_neuron}"

            # Check if both groups exist in the H5 file
            if full_grp_name in h5f and reduced_grp_name in h5f:
                full_grp = h5f[full_grp_name]
                reduced_grp = h5f[reduced_grp_name]

                # Attempt to load the required datasets
                try:
                    # Adjust dataset names here if your CSVs used slightly different headers (e.g. 'ln_value')
                    ln_yyx_sec = full_grp['ln_yyx_sec'][:]
                    ln_yy_sec = reduced_grp['ln_yy_sec'][:]
                    
                    # Assume runtime is based on the full model run
                    run_duration = full_grp['run_duration_sec'][:]
                    
                    # Calculate TE array
                    # If run counts differ (e.g., due to failed runs), truncate to the minimum length
                    min_len = min(len(ln_yyx_sec), len(ln_yy_sec))
                    te_array = np.mean(ln_yyx_sec[:min_len] - ln_yy_sec[:min_len]
                    
                    te_mean = np.mean(te_array)
                    te_std = np.std(te_array, ddof=1) if min_len > 1 else 0.0
                    mean_runtime = np.mean(run_duration)

                    results.append({
                        "task_id": task_id,
                        "source_group_id": src_group,
                        "source_neuron_id": src_neuron,
                        "target_group_id": tgt_group,
                        "target_neuron_id": tgt_neuron,
                        "te mean (nats per second)": te_mean,
                        "te std": te_std,
                        "mean runtime (second)": mean_runtime
                    })
                except KeyError as e:
                    print(f"Missing expected dataset {e} for task {task_id}. Skipping.")
            else:
                print(f"Missing full or reduced data in H5 for task {task_id}. Skipping.")

    # Save to CSV
    if results:
        results_df = pd.DataFrame(results)
        # Ensure column order matches specifications
        columns_order = [
            "task_id", "source_group_id", "source_neuron_id", 
            "target_group_id", "target_neuron_id", 
            "te mean (nats per second)", "te std", "mean runtime (second)"
        ]
        results_df = results_df[columns_order]
        results_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"Transfer entropy calculations complete. Results saved to: {OUTPUT_CSV_PATH}")
    else:
        print("No valid data pairs found to calculate Transfer Entropy.")

if __name__ == "__main__":
    main()
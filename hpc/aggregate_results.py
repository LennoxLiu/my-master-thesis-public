import os
import pandas as pd
import h5py

# Configuration paths
TASKS_REDUCED_PATH = "hpc/tasks_reduced.csv"
TASKS_FULL_PATH = "hpc/tasks_full.csv"
RESULTS_REDUCED_DIR = "results/hpc_runs-reduced"
RESULTS_FULL_DIR = "results/hpc_runs-full"
OUTPUT_H5_PATH = "results/multi_runs_results.h5"

def encode_for_h5(series):
    """
    Ensures pandas Series data types are compatible with h5py datasets.
    Converts object/string columns to byte strings.
    """
    if series.dtype == 'O':  
        return series.astype('S').values
    return series.values

def main():
    # Load task mappings
    tasks_reduced = pd.read_csv(TASKS_REDUCED_PATH)
    tasks_full = pd.read_csv(TASKS_FULL_PATH)

    with h5py.File(OUTPUT_H5_PATH, 'w') as h5f:
        # Create main groups
        grp_reduced = h5f.create_group("reduced")
        grp_full = h5f.create_group("full")

        # 1. Process Reduced Runs
        print("Packing reduced runs...")
        for _, row in tasks_reduced.iterrows():
            task_id = int(row['task_id'])
            group_id = row['group_id']
            neuron_id = int(row['neuron_id'])

            subgroup_name = f"{group_id}_{neuron_id}"
            csv_path = os.path.join(RESULTS_REDUCED_DIR, f"runs_reduced_{task_id}.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                subgrp = grp_reduced.create_group(subgroup_name)
                
                # Save each column as a dataset within the subgroup
                for col in df.columns:
                    if col == 'run':
                        continue
                    subgrp.create_dataset(col, data=encode_for_h5(df[col]))
            else:
                print(f"File missing, skipping: {csv_path}")

        # 2. Process Full Runs
        print("Packing full runs...")
        for _, row in tasks_full.iterrows():
            task_id = int(row['task_id'])
            src_group = row['source_group_id']
            src_neuron = int(row['source_neuron_id'])
            tgt_group = row['target_group_id']
            tgt_neuron = int(row['target_neuron_id'])

            subgroup_name = f"{src_group}_{src_neuron}_to_{tgt_group}_{tgt_neuron}"
            csv_path = os.path.join(RESULTS_FULL_DIR, f"runs_full_{task_id}.csv")

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                subgrp = grp_full.create_group(subgroup_name)
                
                # Save each column as a dataset within the subgroup
                for col in df.columns:
                    if col == 'run':
                        continue
                    subgrp.create_dataset(col, data=encode_for_h5(df[col]))
            else:
                print(f"File missing, skipping: {csv_path}")

    print(f"\nConsolidation complete. Data saved to: {OUTPUT_H5_PATH}")

if __name__ == "__main__":
    main()
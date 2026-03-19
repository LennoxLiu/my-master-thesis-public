import h5py

def main():
    input_path = "results/multi_runs_results.h5"
    output_path = "results/multi_runs_results_transposed.h5"

    print(f"Processing data from {input_path}...")

    try:
        with h5py.File(input_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                
                # 1. Copy the 'reduced' group exactly as is
                if 'reduced' in f_in:
                    print("Copying 'reduced' group...")
                    f_in.copy('reduced', f_out)
                
                # 2. Transpose and copy the 'full' group
                if 'full' in f_in:
                    print("Transposing names in 'full' group...")
                    full_in = f_in['full']
                    full_out = f_out.create_group('full')
                    
                    for key in full_in.keys():
                        if "_to_" in key:
                            # Split and swap the source and target parts
                            parts = key.split("_to_")
                            if len(parts) == 2:
                                new_name = f"{parts[1]}_to_{parts[0]}"
                                # Copy the subgroup and all its datasets to the new file with the new name
                                full_in.copy(key, full_out, name=new_name)
                            else:
                                # Fallback if the naming format is unexpected
                                full_in.copy(key, full_out)
                        else:
                            # Fallback if the delimiter is missing
                            full_in.copy(key, full_out)

        print(f"Successfully saved transposed results to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
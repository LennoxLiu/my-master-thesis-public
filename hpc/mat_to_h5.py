import scipy.io
import h5py
import numpy as np
import os

def convert_mat_to_h5(mat_path, h5_path):
    """
    Converts MATLAB sortedData to HDF5.
    Column 2 -> event_times (float array)
    Column 4 -> group_id
    Row Index -> neuron_id
    """
    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} not found.")
        return

    # Load MATLAB data
    mat_contents = scipy.io.loadmat(mat_path)
    
    if 'sortedData' not in mat_contents:
        print("Error: 'sortedData' variable not found in the .mat file.")
        return

    # sortedData is typically an object array (cell array in MATLAB)
    sorted_data = mat_contents['sortedData']
    
    # Handle cases where MATLAB saves as a 1x1 array containing the table
    if sorted_data.ndim == 2 and sorted_data.shape[0] == 1 and sorted_data.shape[1] == 1:
        sorted_data = sorted_data[0, 0]

    with h5py.File(h5_path, 'w') as h5f:
        for i in range(len(sorted_data)):
            row = sorted_data[i]
            
            # 1. Extract Group ID (Column 4 / Index 3)
            # Handles MATLAB's tendency to wrap strings in nested arrays
            group_val = row[3]
            if isinstance(group_val, np.ndarray):
                group_id = str(group_val.item()).strip()
            else:
                group_id = str(group_val).strip()
            
            if not group_id:
                group_id = "unassigned"

            # 2. Extract Event Times (Column 2 / Index 1)
            event_val = row[1]
            
            # Convert to float array
            event_times = np.array(event_val).flatten().astype(np.float32)

            # 3. Assign unique neuron_id based on row index
            neuron_id = f"{i:03d}"

            # 4. Save to HDF5 structure: group_id/neuron_id
            if group_id not in h5f:
                grp = h5f.create_group(group_id)
            else:
                grp = h5f[group_id]
                
            grp.create_dataset(neuron_id, data=event_times, compression="gzip")

    print(f"Conversion finished. Created {h5_path} with {len(sorted_data)} entries.")

# Execution
convert_mat_to_h5('data/testFile.mat', 'data/event_times_data.h5')
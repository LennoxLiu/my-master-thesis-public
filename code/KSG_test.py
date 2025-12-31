from contextlib import redirect_stdout
from typing import List
from NPEET import mi
from piecewise_lognormal import simulate_processes, compute_reference
import numpy as np
import time

def prepare_data(
    event_time: List[np.ndarray], 
    history_length: int, 
    total_time: float,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares historical sequence data from event times using NumPy.

    This function processes event times from a target and a source process to create
    input features and corresponding targets for a sequence model.

    Args:
        event_time (List[np.ndarray]): A list containing two NumPy arrays:
            - event_time[0]: Timestamps for the target process.
            - event_time[1]: Timestamps for the source process.
        configs (Dict): A dictionary with configuration parameters, including:
            - "history_length" (int): The number of past events to use as history.
            - "total_time" (float): The maximum time to consider for events.
            - "verbose" (bool): If True, prints status messages.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
        - x_history (np.ndarray): History from the source process. 
          Shape: (num_samples, history_length).
          The last column is the time since the last source event.
        - y_present (np.ndarray): The target inter-event times to be predicted.
          Shape: (num_samples,).
        - y_history (np.ndarray): History from the target process.
          Shape: (num_samples, history_length).
    """
    if verbose:
        print('Preparing data with NumPy...')

    # --- 1. Input Validation and Initialization ---
    assert isinstance(event_time, list), "event_time must be a list."
    assert len(event_time) == 2, "This function requires two event time arrays (target and source)."
    assert all(isinstance(evt, np.ndarray) for evt in event_time), "All elements in event_time must be NumPy arrays."


    # Filter events up to the total time
    target_times = event_time[0][event_time[0] < total_time]
    source_times = event_time[1][event_time[1] < total_time]

    # Calculate inter-event times (time differences between consecutive events)
    target_inter_times = np.diff(target_times)
    source_inter_times = np.diff(source_times)

    # Lists to store the generated sequences
    histories_target_list = []
    histories_source_list = []
    time_deltas_list = []
    targets_list = []

    # --- 2. Sequence Generation Loop ---
    source_idx = 0
    # Iterate through each potential target event, starting from where a full history is available
    for target_idx in range(history_length, len(target_times) - 1):
        current_target_time = target_times[target_idx]

        # Find the most recent source event before the current target event
        while source_idx + 1 < len(source_times) and source_times[source_idx + 1] < current_target_time:
            source_idx += 1
            
        # Ensure we have enough historical data from both the source and target processes
        # We need source_idx >= history_length to have enough source inter-event times
        # We also need source_idx < len(source_inter_times) to ensure the slice doesn't go out of bounds
        if source_idx >= history_length and source_idx < len(source_inter_times):
            # This is a valid data point, so we construct the sample
            
            # a) The value to predict (the next inter-event time for the target)
            target_val = target_inter_times[target_idx]
            targets_list.append(target_val)

            # b) The target's own history
            history_target = target_inter_times[target_idx - history_length : target_idx]
            histories_target_list.append(history_target)

            # c) The source's history (full history_length items)
            history_source = source_inter_times[source_idx - history_length + 1 : source_idx + 1]
            histories_source_list.append(history_source)
            
            # d) The time elapsed since the last source event, a critical feature
            time_since_last_source = current_target_time - source_times[source_idx]
            assert time_since_last_source >= 0, "Time delta must be non-negative."
            time_deltas_list.append(time_since_last_source)

    if not targets_list:
        raise ValueError("Could not generate any valid sequences. Try adjusting 'history_length' or 'total_time'.")

    # --- 3. Final Array Conversion ---
    # Convert lists to NumPy arrays
    y_present = np.array(targets_list)
    y_history = np.stack(histories_target_list)
    
    # For x_history, combine the source inter-event times with the time delta feature
    history_source_part = np.stack(histories_source_list)
    time_deltas = np.array(time_deltas_list).reshape(-1, 1) # Reshape for concatenation
    
    x_history = np.hstack([history_source_part[:, :-1], time_deltas])

    return x_history, y_present, y_history

if __name__ == "__main__":
    # Define simulation parameters
    SIMULATION_TIME = 15*60  # seconds
    LAMBDA_X = 30          # events/sec for Poisson process
    seed=67  # For reproducibility

    # The conditional intensity table for process Y
    # The keys are tuples: (is_y_inter_event_time_small, is_x_inter_event_time_small)
    # the tuples are mu and sigma of the log-normal distribution
    intensity_table = {
        (False, False): (-5, 0.5) ,  # Y_t > 10ms, X_t > 10ms, 
        (False, True): (-7, 2),   # Y_t > 10ms, X_t <= 10ms
        (True, False): (-3, 0.5),    # Y_t <= 10ms, X_t > 10ms
        (True, True): (-4, 1.5)      # Y_t <= 10ms, X_t <= 10ms
    }

    # Run the simulation
    print(f"Starting simulation for {SIMULATION_TIME} seconds...")
    x_events, y_events, count_table, plot_data = simulate_processes(SIMULATION_TIME, LAMBDA_X, intensity_table, seed)
    # Compute reference value
    print(count_table)
    with open(f"./results/log_{seed}.txt", 'w') as f:
        with redirect_stdout(f):
            compute_reference(count_table, intensity_table, SIMULATION_TIME)

    for i in range(3):
        print(f"\n--- Estimation Run {i+1} ---")
        
        start = time.perf_counter()
        x_history, y_present, y_history = prepare_data(
            [np.array(y_events), np.array(x_events)],
            history_length=2,
            total_time=SIMULATION_TIME,
            verbose=True
        )
        # Compute conditional mutual information using NPEET
        te_npeet = cmi_estimate = mi(x_history, y_present, y_history, k=3)
        
        elapsed = time.perf_counter() - start
        print(f"prepare_data elapsed: {elapsed:.6f} s")
        print(f"Estimated Transfer Entropy (NPEET): {te_npeet} nats/event")
        te_npeet_sec = te_npeet * len(y_present) / SIMULATION_TIME
        print(f"Estimated Transfer Entropy (NPEET): {te_npeet_sec} nats/sec")
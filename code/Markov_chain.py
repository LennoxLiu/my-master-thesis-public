import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_arrival_times(duration, bin_width, lambda_x, transition_probs, initial_y_prob, seed=None):
    """
    Generates arrival times for two point sequences X and Y.

    Args:
        duration (int): The total duration of the simulation in seconds.
        bin_width (float): The width of each time bin in seconds.
        lambda_x (float): The rate parameter for the Poisson process for X.
        transition_probs (dict): A dictionary representing the transition probabilities for Y.
                                  Key: (Y_t, X_t), Value: P(Y_t+1=1).
        initial_y_prob (float): The initial probability of Y at t=0 being 1.

    Returns:
        tuple: A tuple containing two numpy arrays for the arrival times of X and Y.
    """
    # Calculate the number of bins
    num_bins = int(duration / bin_width)
    
    np.random.seed(seed)

    # Generate X as a Poisson sequence in bins
    x_sequence = np.random.poisson(lambda_x * bin_width, num_bins)
    x_sequence = np.minimum(x_sequence, 1)  # Clamp to 0 or 1 for simplicity

    # Generate Y as a binary sequence (0 or 1) based on X and the previous Y
    y_sequence = np.zeros(num_bins, dtype=int)
    
    # Set the initial state of Y
    y_sequence[0] = 1 if np.random.rand() < initial_y_prob else 0
    
    # Iterate through the bins to generate the Y sequence
    for i in range(1, num_bins):
        prev_y = y_sequence[i - 1]
        prev_x = x_sequence[i - 1]
        
        # Get the probability of Y_t+1=1 from the transition probabilities
        prob_y_is_one = transition_probs[(prev_y, prev_x)]
        
        # Generate the new Y value
        y_sequence[i] = 1 if np.random.rand() < prob_y_is_one else 0
        
    # Calculate arrival times for X and Y using cumulative sum
    time_bins = np.arange(num_bins) * bin_width
    
    x_arrival_times = time_bins[x_sequence == 1]
    y_arrival_times = time_bins[y_sequence == 1]
    
    return x_arrival_times, y_arrival_times, x_sequence, y_sequence


def plot_arrival_times(x_times, y_times, duration, bin_width):
    """
    Plots the arrival times of the X and Y events on a timeline.
    
    Args:
        x_times (np.array): Array of arrival times for X events.
        y_times (np.array): Array of arrival times for Y events.
        duration (int): Total duration of the simulation.
        bin_width (float): The width of each time bin.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot X events as vertical lines
    for t in x_times:
        plt.axvline(x=t, ymin=0.55/3, ymax=1.45/3, color='blue', linewidth=2, label='X Events' if t == x_times[0] else "")
    
    # Plot Y events as vertical lines
    for t in y_times:
        plt.axvline(x=t, ymin=1.55/3, ymax=2.45/3, color='red', linewidth=2, label='Y Events' if t == y_times[0] else "")
    
    plt.yticks([1, 2], ['X Events', 'Y Events'])
    plt.ylim(0.5, 2.5)
    plt.title('Timeline of X and Y Events')
    plt.xlabel('Time (seconds)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # --- Configurable Parameters ---
    total_duration = 1   # in seconds
    bin_width = 1e-3      # in seconds
    lambda_x = 30         # events per second
    initial_y_prob = 0.5  # initial probability of Y at t=0 being 1

    # Store the transition probabilities in a dictionary for easy lookup
    # Key: (Y_t, X_t), Value: P(Y_t+1=1)
    transition_probs = {
        (1, 1): 0.1,
        (1, 0): 0.01,
        (0, 1): 0.6,
        (0, 0): 0.03
    }

    initial_y_prob = 0.5 # Probability of Y at t=0 being 1

    # Generate the sequences and arrival times
    x_arrival_times, y_arrival_times, x_seq, y_seq = generate_arrival_times(
        duration=total_duration,
        bin_width=bin_width,
        lambda_x=lambda_x,
        transition_probs=transition_probs,
        initial_y_prob=initial_y_prob,
        seed=42  # For reproducibility
    )

    print("--- Simulation Parameters ---")
    print(f"Total Duration: {total_duration} seconds")
    print(f"Bin Width: {bin_width} seconds")
    print(f"Poisson Rate (lambda) for X: {lambda_x} events/sec")
    print(f"Number of Bins: {int(total_duration / bin_width)}")
    print("\n--- Event Sequences (First 10 bins) ---")
    print(f"X sequence: {x_seq[:10]}")
    print(f"Y sequence: {y_seq[:10]}")
    
    print("\n--- Arrival Times ---")
    print(f"Number of X events: {len(x_arrival_times)}")
    print(f"First 10 X arrival times: {x_arrival_times[:10]}")
    print("\n")
    print(f"Number of Y events: {len(y_arrival_times)}")
    print(f"First 10 Y arrival times: {y_arrival_times[:10]}")
    
    print("\n--- Summary ---")
    print(f"Total X events generated: {np.sum(x_seq)}")
    print(f"Total Y events generated: {np.sum(y_seq)}")

    # Plot the arrival times
    # plot_arrival_times(x_arrival_times, y_arrival_times, total_duration, bin_width)

    # Entropy and TE of point process (NOT event times)
    # count X_t and Y_t
    counts = {
        (0, 0): 0,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0
    }
    
    toltal_length = len(x_seq) - 1
    for i in range(toltal_length):
        counts[(y_seq[i], x_seq[i])] += 1
    
    # p(Y_t+1=1|Y_t,X_t) 
    p_yyx = 0
    for (y_t, x_t) in {(0,0), (0,1), (1,0), (1,1)}:
        p_yyx += (counts[(y_t, x_t)] / toltal_length) * transition_probs[(y_t, x_t)]

    # p(Y_t+1=1|Y_t)
    p_yy = (counts[(0,0)] + counts[(1,0)]) / toltal_length * (transition_probs[(0, 0)] + transition_probs[(1,0)]) + \
           (counts[(1,1)] + counts[(0,1)]) / toltal_length * (transition_probs[(1, 1)] + transition_probs[(0,1)])

    H_yy = - (p_yy * np.log(p_yy) + (1 - p_yy) * np.log(1 - p_yy))
    H_yyx = - (p_yyx * np.log(p_yyx) + (1 - p_yyx) * np.log(1 - p_yyx))
    print(f"Estimated H(Y_t+1|Y_t) (nats per time step): {H_yy}")
    print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per time step): {H_yyx}")
    print(f"(Y_t+1|Y_t) : {- p_yy * np.log(p_yy)}")
    print(f"(Y_t+1|Y_t,X_t) : {- p_yyx * np.log(p_yyx)}")
    print(f"(Y_t+1|Y_t)-(Y_t+1|Y_t,X_t) : {- p_yy * np.log(p_yy) + p_yyx * np.log(p_yyx)} nats per event")
    print(f"(Y_t+1|Y_t)-(Y_t+1|Y_t,X_t) : {(- p_yy * np.log(p_yy) + p_yyx * np.log(p_yyx) )* np.sum(y_seq) / total_duration} nats per second")
    TE = H_yy - H_yyx
    print(f"Estimated Transfer Entropy (nats per time step): {TE}")
    print(f"Estimated Transfer Entropy (nats per second): {TE / bin_width}")

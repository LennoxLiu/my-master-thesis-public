from contextlib import redirect_stdout
import time
import numpy as np
import matplotlib.pyplot as plt
from entropy_tpp import TE_estimation_tpp, run_multiple_estimation, save_dict_indented
import torch
from scipy.stats import norm

def simulate_processes(total_time, lambda_x, intensity_table, seed=None):
    """
    Simulates a Poisson process (X) and a conditional stochastic process (Y)
    sequentially.

    X is simulated first as a standard Poisson process using an efficient
    vectorized method.
    Y's intensity then depends on the history of both X and Y, with the
    intensity remaining constant between Y events.

    Args:
        total_time (float): The total duration of the simulation in seconds.
        lambda_x (float): The constant intensity (events/sec) of the Poisson process X.

    Returns:
        tuple: (source_events, target_events, count_table, interval_table)
    """
    if seed is not None:
        np.random.seed(seed)

    # --- Step 1: Simulate Process X (Poisson) ---
    # This is done by drawing a large number of inter-arrival times at once
    # from an exponential distribution and then computing their cumulative sum.
    # We estimate the total number of events as lambda_x * total_time and add a buffer.
    num_events_x_estimate = int(lambda_x * total_time * 2)  # Add a 100% buffer
    inter_arrival_times_x = np.random.exponential(1.0/lambda_x, num_events_x_estimate)
    
    # Calculate the event times as the cumulative sum of inter-arrival times
    event_times_x_raw = np.cumsum(inter_arrival_times_x)
    
    # Truncate the event times at the total simulation time
    events_x = event_times_x_raw[event_times_x_raw <= total_time].tolist()

    assert event_times_x_raw[-1] > total_time

    # --- Step 2: Simulate Process Y (Conditional) ---
    # The intensity for Y is constant between Y events.
    events_y = []
    current_time = 0
    
    # Track the last two event times for Y to calculate inter-event times.
    last_event_y = -np.inf
    second_last_event_y = -np.inf  # No second last event at the start
    
    # Use an index to efficiently track the position in the events_x list
    x_event_index = 0


    count_table = {
        (False, False): 0,
        (False, True): 0,
        (True, False): 0,
        (True, True): 0
    }
    interval_table = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): []
    }

    while current_time < total_time:
        # Advance the index to find the most recent X event(s) that occurred
        # before or at the current time. This is much more efficient than searching the list.
        while x_event_index < len(events_x) and events_x[x_event_index] <= current_time:
            x_event_index += 1
            
        # Get the last two event times for X using the index
        last_event_x = events_x[x_event_index - 1] if x_event_index > 0 else -np.inf
        second_last_event_x = events_x[x_event_index - 2] if x_event_index > 1 else -np.inf
        
        # Calculate inter-event times
        inter_event_x = last_event_x - second_last_event_x if second_last_event_x > -np.inf else np.inf
        inter_event_y = last_event_y - second_last_event_y if second_last_event_y > -np.inf else np.inf

        # Determine the correct lambda_y from the table
        y_small = inter_event_y <= 0.01  # 10ms
        x_small = inter_event_x <= 0.01

        lambda_y = intensity_table[(y_small, x_small)]
        count_table[(y_small, x_small)] += 1 # Count how many times we are in this state for entropy calculation

        # Simulate the time to the next Y event using an exponential distribution
        time_to_next_y = np.random.exponential(1.0 / lambda_y)
        # print(f"Current time: {current_time:.5f}, λ_y: {lambda_y:.5f}, "
        #       f"Time to next Y: {time_to_next_y:.5f}, ")
        next_event_y_time = last_event_y + time_to_next_y if last_event_y > 0 else time_to_next_y

        interval_table[(y_small, x_small)].append(time_to_next_y)
        if next_event_y_time <= total_time:
            events_y.append(next_event_y_time)
            # Update the last two Y event times
            second_last_event_y = last_event_y
            last_event_y = next_event_y_time
            # Advance time to the new Y event
            current_time = next_event_y_time
        else:
            # The next event is past the simulation time, so we stop.
            break
    
    return events_x, events_y, count_table, interval_table

from scipy.stats import expon

# def monte_carlo_entropy(scale1, scale2, alpha, data, n_samples=100000):
#     """
#     Calculates the differential entropy of a mixture of two exponential distributions
#     using a Monte Carlo approximation.

#     The differential entropy H(X) is defined as: H(X) = -E[log(f(X))],
#     where f(X) is the probability density function (PDF) of the mixture.
#     The Monte Carlo method approximates this expectation with an average over samples:
#     H(X) ≈ -(1/N) * sum(log(f(x_i))) for i=1 to N.

#     Args:
#         scale1 (float): The scale parameter (1/rate) for the first exponential component.
#         scale2 (float): The scale parameter (1/rate) for the second exponential component.
#         alpha (float): The mixing weight for the first distribution (0 <= alpha <= 1).
#         n_samples (int): The number of samples to use for the Monte Carlo simulation.
#                          A larger number increases accuracy but also computation time.

#     Returns:
#         float: The estimated differential entropy of the mixture distribution.
#     """
#     # 1. Define the two exponential distributions
#     dist1 = expon(scale=scale1)
#     dist2 = expon(scale=scale2)

#     # 2. Generate samples from the mixture distribution
#     # We use a random choice based on the mixing coefficient 'alpha'
#     is_from_dist1 = np.random.rand(n_samples) < alpha
#     samples = np.zeros(n_samples)
    
#     samples[is_from_dist1] = dist1.rvs(size=np.sum(is_from_dist1))
#     samples[~is_from_dist1] = dist2.rvs(size=np.sum(~is_from_dist1))

#     # Ensure inputs are numpy arrays
#     data = np.asarray(data)
#     samples = np.asarray(samples)

#     # Empirical survival function S(x) = P(sample > x), computed by counting samples larger than each data point
#     # Vectorized count for efficiency: shape (n_samples, n_data) then sum over samples
#     survival_values = (samples[:, None] > data[None, :]).sum(axis=0) / float(n_samples)

#     # Optional: derive an empirical pdf by differentiating the survival curve on sorted data,
#     # then interpolate back to the original data order for use in downstream computations.
#     sorted_idx = np.argsort(data)
#     data_sorted = data[sorted_idx]
#     survival_sorted = survival_values[sorted_idx]

#     # Numerical derivative to estimate pdf: f(x) ≈ -dS/dx
#     pdf_sorted = -np.gradient(survival_sorted, data_sorted, edge_order=2)
#     pdf_sorted = np.maximum(pdf_sorted, 0.0)  # remove small negative numerical artifacts

#     # Interpolate pdf back to original data order
#     empirical_pdf = np.interp(data, data_sorted, pdf_sorted)

#     empirical_pdf = np.clip(empirical_pdf, 1e-30, None)  # Avoid log(0)
#     survival_values = np.clip(survival_values, 1e-30, None)  # Avoid division by zero

#     # 4. Approximate the expected value of -log(f(x))
#     # This is the Monte Carlo approximation of the differential entropy
#     entropy = -np.mean(np.log(empirical_pdf/survival_values))

#     return entropy


def compute_reference(count_table, intensity_table, interval_table, total_duration):
    total_length = sum(count_table.values())
    # p(Y_t+1=1|Y_t,X_t)
    H_yyx = 0
    for (y_t, x_t) in {(False, False), (False, True), (True, False), (True, True)}:
        # data = interval_table[(y_t, x_t)]
        lambda_ = 1/intensity_table[(y_t, x_t)]
        frequency = count_table[(y_t, x_t)] / total_length
        H_yyx += frequency * np.log(lambda_)

    p_y_small = (count_table[(1,0)] + count_table[(1,1)]) / total_length
    p_x_small = (count_table[(0,1)] + count_table[(1,1)]) / total_length

    data_y_small = interval_table[(True,True)] + interval_table[(True,False)]
    data_y_large = interval_table[(False,True)] + interval_table[(False,False)]

    # Probabilities for Y_t
    count_y_small = count_table[(True, False)] + count_table[(True, True)]
    count_y_large = count_table[(False, False)] + count_table[(False, True)]

    # --- H(Y | Y_small) component ---
    H_y_small = 0.0
    if count_y_small > 0:
        # Mixing weight alpha_1 = P(X_small | Y_small)
        alpha_1 = count_table[(True, True)] / count_y_small
        
        # Scales
        scale_1_true = 1.0 / intensity_table[(True, True)]  # State (Y=small, X=small)
        scale_1_false = 1.0 / intensity_table[(True, False)] # State (Y=small, X=large)
        
        H_y_small = analytical_entropy_metric(
            scale1=scale_1_true, 
            scale2=scale_1_false, 
            alpha=alpha_1, 
            data=data_y_small
        )

    # --- H(Y | Y_large) component ---
    H_y_large = 0.0
    if count_y_large > 0:
        # Mixing weight alpha_2 = P(X_small | Y_large)
        alpha_2 = count_table[(False, True)] / count_y_large
        
        # Scales
        scale_2_true = 1.0 / intensity_table[(False, True)]  # State (Y=large, X=small)
        scale_2_false = 1.0 / intensity_table[(False, False)] # State (Y=large, X=large)
        
        H_y_large = analytical_entropy_metric(
            scale1=scale_2_true, 
            scale2=scale_2_false, 
            alpha=alpha_2, 
            data=data_y_large
        )

    H_yy = p_y_small * H_y_small + (1 - p_y_small) * H_y_large
    # print(f"- Estimated H(Y_t+1|Y_t) (nats per event): {H_yy}")
    # print(f"- Estimated H(Y_t+1|Y_t,X_t) (nats per event): {H_yyx}")
    print(f"- Estimated H(Y_t+1|Y_t) (nats per second): {H_yy * total_length / total_duration:.5f}")
    print(f"- Estimated H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx * total_length / total_duration:.5f}")

    TE = H_yy - H_yyx
    # print(f"- Estimated Transfer Entropy (nats per event): {TE}")
    print(f"- Estimated Transfer Entropy (nats per second): {TE * total_length / total_duration:.5f}")

    return H_yy, H_yyx


def analytical_entropy_metric(scale1, scale2, alpha, data):
    """
    Calculates the analytical expectation E[-log(h(T))] for a mixture
    of two exponential distributions, evaluated over the provided data.

    This metric E[-log(h(T))] = -E[log(f(T)/S(T))] is used as a component
    of the transfer entropy calculation.

    Args:
        scale1 (float): The scale parameter (1/rate) for the first exponential.
        scale2 (float): The scale parameter (1/rate) for the second exponential.
        alpha (float): The mixing weight for the first distribution (0 <= alpha <= 1).
        data (list or np.array): The observed inter-arrival times (T) from the
                                 simulation, over which to compute the expectation.

    Returns:
        float: The computed metric E[-log(h(T))].
    """
    # Ensure data is a numpy array for vectorized operations
    t = np.asarray(data)
    if t.size == 0:
        return 0.0  # Handle case with no data to avoid errors

    # Convert scales (beta) to rates (lambda)
    lambda1 = 1.0 / scale1
    lambda2 = 1.0 / scale2

    # 1. Calculate the PDF components
    f1_t = lambda1 * np.exp(-lambda1 * t)
    f2_t = lambda2 * np.exp(-lambda2 * t)
    
    # 2. Calculate the Survival components
    s1_t = np.exp(-lambda1 * t)
    s2_t = np.exp(-lambda2 * t)

    # 3. Calculate the mixture PDF and Survival
    pdf_mixture = alpha * f1_t + (1 - alpha) * f2_t
    survival_mixture = alpha * s1_t + (1 - alpha) * s2_t

    # 4. Clip for numerical stability (avoid log(0) or division by zero)
    pdf_mixture = np.clip(pdf_mixture, 1e-30, None)
    survival_mixture = np.clip(survival_mixture, 1e-30, None)

    # 5. Calculate the hazard function h(t) = f(t) / S(t)
    hazard = pdf_mixture / survival_mixture

    # 6. Calculate the metric E[-log(h(T))]
    # The expectation is approximated by the mean over the provided data
    entropy_metric = -np.mean(np.log(hazard))

    return entropy_metric


if __name__ == "__main__":
    # Define simulation parameters
    SIMULATION_TIME = 5*60  # seconds
    LAMBDA_X = 30          # events/sec for Poisson process
    seed=43  # For reproducibility

    # The conditional intensity table for process Y
    # The keys are tuples: (is_y_inter_event_time_small, is_x_inter_event_time_small)
    intensity_table = {
        (False, False): 0.03 / 1e-3,  # Y_t > 10ms, X_t > 10ms,
        (False, True): 0.3 / 1e-3,   # Y_t > 10ms, X_t <= 10ms
        (True, False): 0.06 / 1e-3,    # Y_t <= 10ms, X_t > 10ms
        (True, True): 0.1 / 1e-3      # Y_t <= 10ms, X_t <= 10ms
    }

    # Run the simulation
    print(f"Starting simulation for {SIMULATION_TIME} seconds...")
    x_events, y_events, count_table, interval_table = simulate_processes(SIMULATION_TIME, LAMBDA_X, intensity_table, seed)
    compute_reference(count_table, intensity_table, interval_table,SIMULATION_TIME)
    print(f"Simulation completed. Generated {len(x_events)} events for X and {len(y_events)} events for Y.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    time_series_length = SIMULATION_TIME   # in seconds, Length of the time series
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    configs = {
        "model_config_yy": {
            "model_name": "LogNormMix_yy",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 8,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 64,        # 16 Number of components for a mixture model
            "hidden_sizes": [32, 32],     # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "model_config_yyx": {
            "model_name": "LogNormMix_yyx",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 16,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 64,  # 16 Number of components for a mixture model
            "hidden_sizes": [32, 32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "train_config_yy": {
            "L2_weight": 1e-4,          # L2 regularization parameter
            # "L_entropy_weight": 1e-1,      # Weight for the entropy regularization term
            # "L_sep_weight": 1e-5,               # Weight for the separation regularization term
            "L_scale_weight": 1e-1,             # Weight for the scale regularization term
            # "L_mean_match_weight": 1,        # Weight for the mean matching regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 500,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 20,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "train_config_yyx": {
            "L2_weight": 1e-4,          # L2 regularization parameter
            # "L_entropy_weight": 1e-1,      # Weight for the entropy regularization term
            # "L_sep_weight": 1e-5,               # Weight for the separation regularization term
            "L_scale_weight": 1e-1,             # Weight for the scale regularization term
            # "L_mean_match_weight": 1,        # Weight for the mean matching regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 500,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 20,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "data_prep_config":{
            "batch_size": 128,          # Number of sequences in a batch
            "shuffle": False,                 # Whether to shuffle the time series before splitting into train/val/test
            "total_time": time_series_length,              # in second, Total time of the sequences
            "verbose": False
        },
        "device": device,
        "verbose": False,  # Whether to print the training statistics
        "plot_histograms": True,  # Whether to plot the conditional histograms
        "plot_pp": False,            # Whether to plot the probability - probability plots
        "history_length": 8,             # in number of bins, Length of the history to use for the model
        "num_mc_samples": 20480,        # Number of Monte Carlo samples for entropy estimation
    }
    with open(f"./results/log_{seed}.txt", 'w') as f:
        with redirect_stdout(f):
            compute_reference(count_table, intensity_table, interval_table,SIMULATION_TIME)

    save_dict_indented(configs, f"./results/config_{seed}.txt")
    
    start_time = time.time()
    (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
            event_time=[torch.tensor(y_events), torch.tensor(x_events)], 
            configs=configs, 
            seed=seed
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- TE Estimation Completed in {duration/60:.2f} minutes ---")
    print(f"Estimated Transfer Entropy (nats per second): {TE_test:.5f}")
    print(f"Estimated H(Y_t+1|Y_t) (nats per second): {H_yy_test:.5f}")
    print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx_test:.5f}")
    print(f"Log Loss H(Y_t+1|Y_t): {log_loss_yy:.5f}")
    print(f"Log Loss H(Y_t+1|Y_t,X_t): {log_loss_yyx:.5f}")

    run_multiple_estimation(
        target_events=torch.tensor(y_events),
        source_events=torch.tensor(x_events),
        configs=configs,
        n_runs=10,
        seed=seed
    )
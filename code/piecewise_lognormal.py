import multiprocessing
import os
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from entropy_tpp import TE_estimation_tpp, run_multiple_estimation
import torch
from numpy.polynomial.laguerre import laggauss
import time
from scipy.stats import lognorm, norm
from numpy.polynomial.hermite import hermgauss
import math
from entropy_tpp import save_dict_indented
from contextlib import redirect_stdout

def lognormal_entropy(mu,sigma):
    """
    Computes the entropy of a log-normal distribution given its parameters.

    Args:
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.

    Returns:
        float: The entropy of the log-normal distribution.
    """
    # Compute the entropy using the formula for log-normal entropy
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2) + mu

def ghq_entropy_lognormal_mixture(mu1, sigma1, mu2, sigma2, alpha, degree=256):
    """
    Calculates the differential entropy of a two-component log-normal mixture
    using Gauss-Hermite Quadrature.

    This is a deterministic and more accurate alternative to monte_carlo_entropy.

    Args:
        mu1, sigma1: Parameters of the underlying normal for the first component.
        mu2, sigma2: Parameters of the underlying normal for the second component.
        alpha (float): The mixing weight for the first component (0 <= alpha <= 1).
        degree (int): The number of points for the quadrature.

    Returns:
        float: The calculated differential entropy of the mixture distribution.
    """
    # 1. Get Gauss-Hermite roots (t_k) and weights (w_k)
    t_k, w_k = hermgauss(degree)

    # 2. Define the two log-normal distributions and the full mixture PDF
    dist1 = lognorm(s=sigma1, scale=np.exp(mu1))
    dist2 = lognorm(s=sigma2, scale=np.exp(mu2))

    def mixture_pdf(z):
        # Add a small epsilon for numerical stability to prevent log(0)
        return alpha * dist1.pdf(z) + (1 - alpha) * dist2.pdf(z) + 1e-30

    # 3. Calculate the first expectation: E_{Z1}[-log p_Z(Z1)]
    # Change of variables for GHQ: y_k = μ + sqrt(2)*σ*t_k
    y1_k = mu1 + math.sqrt(2) * sigma1 * t_k
    # Transform to log-normal scale
    z1_k = np.exp(y1_k)
    # Evaluate the integrand
    integrand1 = -np.log(mixture_pdf(z1_k))
    # Compute the integral using the GHQ formula
    integral1 = (1.0 / math.sqrt(math.pi)) * np.sum(w_k * integrand1)

    # 4. Calculate the second expectation: E_{Z2}[-log p_Z(Z2)]
    y2_k = mu2 + math.sqrt(2) * sigma2 * t_k
    z2_k = np.exp(y2_k)
    integrand2 = -np.log(mixture_pdf(z2_k))
    integral2 = (1.0 / math.sqrt(math.pi)) * np.sum(w_k * integrand2)

    # 5. Combine the results based on the mixture weights
    total_entropy = alpha * integral1 + (1 - alpha) * integral2
    
    return total_entropy

def exp_sinh_nodes_weights(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates nodes and weights for Exp-Sinh Quadrature on (0, inf) using NumPy.

    x = exp( (pi/2) * sinh(t) )
    dx/dt = (pi/2) * cosh(t) * exp( (pi/2) * sinh(t) )

    The integral is approximated by a sum using the trapezoidal rule on the
    transformed function, which converges extremely quickly.

    Args:
        degree (int): The number of points to use (half positive, half negative).

    Returns:
        A tuple containing:
        - nodes (np.ndarray): The evaluation points `x_k` in (0, inf).
        - weights (np.ndarray): The corresponding weights `w_k`.
    """
    # Use float64 for precision, matching NumPy's default
    h = 8.0 / (2 * degree)
    k = np.arange(-degree, degree + 1, dtype=np.float64)

    # Discretize the transformed variable `t`
    t_nodes = k * h

    # The transformation for the nodes is x_k = exp( (pi/2) * sinh(t_k) )
    pi_half_sinh_t = 0.5 * np.pi * np.sinh(t_nodes)
    nodes = np.exp(pi_half_sinh_t)

    # The weights are h * dx/dt evaluated at t_k
    pi_half_cosh_t = 0.5 * np.pi * np.cosh(t_nodes)
    weights = h * pi_half_cosh_t * np.exp(pi_half_sinh_t)

    return nodes, weights


def calculate_entropy_exp_sinh(
    pdf_func: Callable,
    nodes: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    Calculates entropy for a distribution using numerical quadrature.

    This function calculates the entropy integral directly:
    H(Z) = -E[log p(Z)] = -Integral[ p(z) * log(p(z)) dz ]

    Args:
        pdf_func (Callable): The probability density function p(z) of the distribution.
        nodes (np.ndarray): Nodes for quadrature `x_k` from (0, inf).
        weights (np.ndarray): Weights for quadrature.

    Returns:
        The differential entropy of the distribution.
    """
    
    # 1. Evaluate the PDF at the nodes
    # Add a small epsilon for numerical stability to prevent log(0)
    prob_x = pdf_func(nodes) + 1e-30

    # 2. Evaluate the components of the integrand
    log_prob_x = np.log(prob_x)
    
    # 3. Form the complete integrand: -p(x) * log(p(x))
    integrand = prob_x * (-log_prob_x)

    # 4. Calculate the integral by taking the weighted sum
    # The weights from exp_sinh_nodes_weights already include the h*dx/dt term
    entropy = np.sum(integrand * weights)

    return entropy


def esq_entropy_lognormal_mixture(
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    alpha: float,
    degree: int = 256
) -> float:
    """
    Calculates the differential entropy of a two-component log-normal mixture
    using Exp-Sinh Quadrature (ESQ) with NumPy.

    Args:
        mu1, sigma1: Parameters of the underlying normal for the first component.
        mu2, sigma2: Parameters of the underlying normal for the second component.
        alpha (float): The mixing weight for the first component (0 <= alpha <= 1).
        degree (int): The number of points for the quadrature.

    Returns:
        float: The calculated differential entropy of the mixture distribution.
    """
    
    # 1. Get Exp-Sinh nodes and weights for the interval (0, inf)
    nodes, weights = exp_sinh_nodes_weights(degree)

    # 2. Define the two log-normal distributions and the full mixture PDF
    # Scipy's lognorm takes s=sigma (shape) and scale=exp(mu)
    dist1 = lognorm(s=sigma1, scale=np.exp(mu1))
    dist2 = lognorm(s=sigma2, scale=np.exp(mu2))

    def mixture_pdf(z):
        return alpha * dist1.pdf(z) + (1 - alpha) * dist2.pdf(z)

    # 3. Calculate the entropy using the generic quadrature function
    # This directly computes H(Z) = -Integral[p(z) * log(p(z)) dz]
    entropy = calculate_entropy_exp_sinh(mixture_pdf, nodes, weights)

    # 4. Return the result as a scalar float
    return entropy

def monte_carlo_entropy(mu1, sigma1, mu2, sigma2, alpha, n_samples=100000):
    dist1 = lognorm(s=sigma1, scale=np.exp(mu1))
    dist2 = lognorm(s=sigma2, scale=np.exp(mu2))
    is_from_dist1 = np.random.rand(n_samples) < alpha
    samples = np.zeros(n_samples)
    samples[is_from_dist1] = dist1.rvs(size=np.sum(is_from_dist1))
    samples[~is_from_dist1] = dist2.rvs(size=np.sum(~is_from_dist1))
    mixture_pdf_values = alpha * dist1.pdf(samples) + (1 - alpha) * dist2.pdf(samples)
    epsilon = 1e-30
    mixture_pdf_values = mixture_pdf_values + epsilon
    entropy = -np.mean(np.log(mixture_pdf_values))
    return entropy


def gauss_laguerre_entropy(mu1, sigma1, mu2, sigma2, alpha, degree=128):
    """
    Calculates the differential entropy of a log-normal mixture using Gauss-Laguerre quadrature.

    This method directly approximates the entropy integral: H(X) = -∫f(x)log(f(x))dx.
    It transforms the integral into the form ∫g(x)e^(-x)dx required by the quadrature,
    providing a highly accurate and deterministic result.

    Args:
        mu1 (float): Mean of the underlying normal distribution for the first component.
        sigma1 (float): Standard deviation for the first component.
        mu2 (float): Mean of the underlying normal distribution for the second component.
        sigma2 (float): Standard deviation for the second component.
        alpha (float): The mixing weight for the first distribution (0 <= alpha <= 1).
        degree (int): The number of points for the quadrature. A higher degree
                      increases accuracy. 100 is often a very robust choice.

    Returns:
        float: The calculated differential entropy of the mixture distribution.
    """
    # 1. Get the Gauss-Laguerre roots (x_i) and weights (w_i)
    # These are the specific points and weights for the numerical integration.
    x_i, w_i = laggauss(degree)

    # 2. Define the two log-normal distributions from the input parameters
    dist1 = lognorm(s=sigma1, scale=np.exp(mu1))
    dist2 = lognorm(s=sigma2, scale=np.exp(mu2))

    # 3. Calculate the mixture PDF, f(x), at each of the Laguerre points x_i
    f_values = alpha * dist1.pdf(x_i) + (1 - alpha) * dist2.pdf(x_i)

    # Add a small constant for numerical stability to prevent log(0)
    epsilon = 1e-30
    log_f_values = np.log(f_values + epsilon)

    # 4. Define and evaluate the function g(x) for the quadrature
    # To match the form ∫g(x)e^(-x)dx, our g(x) must be f(x)log(f(x))e^x
    g_values = f_values * log_f_values * np.exp(x_i)

    # 5. Calculate the integral by taking the dot product of weights and g(x_i) values
    # The entropy H(X) is the *negative* of this integral.
    entropy = -np.dot(w_i, g_values)

    return entropy

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
        tuple: A tuple containing two lists: (events_x, events_y)
               where each list contains the timestamps of the events for
               the respective process.
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

    plot_data = {
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
        y_small = inter_event_y <= 0.01 # 10ms
        x_small = inter_event_x <= 0.01
        
        mu, sigma = intensity_table[(y_small, x_small)]
        count_table[(y_small, x_small)] += 1 # Count how many times we are in this state for entropy calculation

        # Simulate the time to the next Y event using an log-normal distribution
        time_to_next_y = np.random.lognormal(mean=mu, sigma=sigma)
        # print(f"Current time: {current_time:.4f}, mean: {mu:.4f}, sigma: {sigma:.4f} "
        #       f"Time to next Y: {time_to_next_y:.4f}")
        next_event_y_time = last_event_y + time_to_next_y if last_event_y > 0 else time_to_next_y
        
        # Store data for plotting
        plot_data[(y_small, x_small)].append(time_to_next_y)

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
    
    return events_x, events_y, count_table, plot_data

# input: count_table, intensity_table, total_duration
def compute_reference(count_table, intensity_table, total_duration):
    total_length = sum(count_table.values())
    # p(Y_t+1=1|Y_t,X_t)
    H_yyx = 0
    for (y_t, x_t) in {(0,0), (0,1), (1,0), (1,1)}:
        frequency = count_table[(y_t, x_t)] / total_length
        H_yyx += frequency * lognormal_entropy(*intensity_table[(y_t, x_t)]) if frequency > 0 else 0

    p_y_small = (count_table[(1,0)] + count_table[(1,1)]) / total_length
    p_x_small = (count_table[(0,1)] + count_table[(1,1)]) / total_length

    print("\n --- Gauss-Hermite Entropy Reference Calculation ---")
    H_y_small = ghq_entropy_lognormal_mixture( *intensity_table[(True, True)],*intensity_table[(True, False)], p_x_small, degree=256)
    H_y_large = ghq_entropy_lognormal_mixture( *intensity_table[(False, True)],*intensity_table[(False, False)], p_x_small, degree=256)
    H_yy = p_y_small * H_y_small + (1 - p_y_small) * H_y_large
    print(f"- [GHQ]Reference H(Y_t+1|Y_t) (nats per event): {H_yy}")
    print(f"- [GHQ]Reference H(Y_t+1|Y_t) (nats per second): {H_yy* total_length / total_duration}")
    TE = H_yy - H_yyx
    print(f"- [GHQ]Reference Transfer Entropy (nats per event): {TE}")
    print(f"- [GHQ]Reference Transfer Entropy (nats per second): {TE * total_length / total_duration}")

    print("\n --- Exp-Sinh Entropy Reference Calculation ---")
    H_y_small_esq = esq_entropy_lognormal_mixture( *intensity_table[(True, True)],*intensity_table[(True, False)], p_x_small, degree=5120)
    H_y_large_esq = esq_entropy_lognormal_mixture( *intensity_table[(False, True)],*intensity_table[(False, False)], p_x_small, degree=5120)
    H_yy_esq = p_y_small * H_y_small_esq + (1 - p_y_small) * H_y_large_esq
    print(f"- [ESQ]Reference H(Y_t+1|Y_t) (nats per event): {H_yy_esq}")
    print(f"- [ESQ]Reference H(Y_t+1|Y_t) (nats per second): {H_yy_esq* total_length / total_duration}")
    TE_esq = H_yy_esq - H_yyx
    print(f"- [ESQ]Reference Transfer Entropy (nats per event): {TE_esq}")
    print(f"- [ESQ]Reference Transfer Entropy (nats per second): {TE_esq * total_length / total_duration}")
   

    print("\n --- Monte Carlo Entropy Reference Calculation ---")
    H_y_small = monte_carlo_entropy( *intensity_table[(True, True)],*intensity_table[(True, False)], p_x_small)
    H_y_large = monte_carlo_entropy( *intensity_table[(False, True)],*intensity_table[(False, False)], p_x_small)
    H_yy = p_y_small * H_y_small + (1 - p_y_small) * H_y_large
    print(f"- [MC]Reference H(Y_t+1|Y_t) (nats per event): {H_yy}")
    print(f"- [MC]Reference H(Y_t+1|Y_t) (nats per second): {H_yy* total_length / total_duration}")
    # print(f"- [MC]Reference H(Y_t+1|Y_t,X_t) (nats per event): {H_yyx}")
    # print(f"- [MC]Reference H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx* total_length / total_duration}")
    TE = H_yy - H_yyx
    print(f"- [MC]Reference Transfer Entropy (nats per event): {TE}")
    print(f"- [MC]Reference Transfer Entropy (nats per second): {TE * total_length / total_duration}")

    print(f"- Reference H(Y_t+1|Y_t,X_t) (nats per event): {H_yyx}")
    print(f"- Reference H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx* total_length / total_duration}")

    H_yy_sec = H_yy_esq * total_length / total_duration
    H_yyx_sec = H_yyx * total_length / total_duration
    return H_yy_sec, H_yyx_sec, H_yy_sec - H_yyx_sec


def plot_conditional_histograms(plot_data: dict, seed: int, bins: int = 100):
    fig, axes = plt.subplots(4, 1, figsize=(8, 22), sharex=False, sharey=False)
    fig.suptitle("Simulated data", fontsize=16)

    conditions= {
        (True, True): "SS",
        (True, False): "SL",
        (False, True): "LS",
        (False, False): "LL"
    }

    x_maxes = { "LL": 0.05, "LS": 0.2, "SL": 0.3, "SS": 0.5 }
    for ax, (key, values) in zip(axes, plot_data.items()):
        ax.set_xlim(0, x_maxes[conditions[key]])
        x_values = np.array(values)
        ax.hist(x_values[x_values<x_maxes[conditions[key]]], bins=bins, color='lightgreen', alpha=0.7,density=True,label=f"Samples: {len(x_values)}, Max: {np.max(x_values):.4f}")
        ax.set_ylabel(conditions[key])
        ax.grid(True)
        ax.legend()
        # print(f"Condition {conditions[key]}: max={np.max(values):.4f}")


    axes[-1].set_xlabel('Inter-event Time (seconds)', fontsize=14)
    fig.text(0.02, 0.5, 'Counts', va='center', rotation='vertical', fontsize=14)


    os.makedirs(f"./results/hists/", exist_ok=True)
    # Delete all files in the directory before saving
    hist_dir = "./results/hists/"
    if os.path.exists(hist_dir):
        for filename in os.listdir(hist_dir):
            file_path = os.path.join(hist_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    
    plt.savefig(f"./results/hists/cond_hist_simu_{seed}.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    # Define simulation parameters
    SIMULATION_TIME = 5*60  # seconds
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
    plot_conditional_histograms(plot_data, seed=seed, bins=100)

    print(count_table)
    with open(f"./results/log_{seed}.txt", 'w') as f:
        with redirect_stdout(f):
            compute_reference(count_table, intensity_table, SIMULATION_TIME)

    # Print summary statistics
    print("\n--- Simulation Results ---")
    print(f"Total events for Process X: {len(x_events)}")
    print(f"Total events for Process Y: {len(y_events)}")
    print(f"Average rate for Process X: {len(x_events) / SIMULATION_TIME:.4f} events/sec")
    print(f"Average rate for Process Y: {len(y_events) / SIMULATION_TIME:.4f} events/sec")
    

    torch.manual_seed(seed)
    np.random.seed(seed)
    time_series_length = SIMULATION_TIME   # in seconds, Length of the time series
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    configs = {
        "model_config_yy": {
            "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 4,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 4,        # 16 Number of components for a mixture model
            "hidden_sizes": [64],     # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "mlp", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "Tanh",
        },
        "model_config_yyx": {
            "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 4,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 4,  # 16 Number of components for a mixture model
            "hidden_sizes": [32, 32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "mlp", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "ReLU",
        },
        "train_config_yy": {
            "L2_weight": 1e-3,          # L2 regularization parameter
            "L_entropy_weight": 1e-4,      # Weight for the entropy regularization term
            "L_sep_weight": 1e-4,               # Weight for the separation regularization term
            "L_scale_weight": 1e-4,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "train_config_yyx": {
            "L2_weight": 1e-3,          # L2 regularization parameter
            "L_entropy_weight": 5e-10,      # Weight for the entropy regularization term
            "L_sep_weight": 2e-4,               # Weight for the separation regularization term
            "L_scale_weight": 1e-9,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
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
        "history_length": 16,             # in number of bins, Length of the history to use for the model
        # "num_mc_samples": 256000,  # Number of Monte Carlo samples for TE estimation
    }

    # Save the config for reference
    save_dict_indented(configs, f"./results/config_{seed}.txt")
    
    # start_time = time.time()
    # (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
    #         event_time=[torch.tensor(y_events), torch.tensor(x_events)], 
    #         configs=configs, 
    #         seed=seed
    # )
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"\n--- TE Estimation Completed in {duration/60:.2f} minutes ---")
    # print(f"Estimated Transfer Entropy (nats per event): {TE_test}")
    # print(f"Estimated Transfer Entropy (nats per second): {TE_test * len(y_events) / SIMULATION_TIME}")
    # print(f"Estimated H(Y_t+1|Y_t) (nats per event): {H_yy_test}")
    # print(f"Estimated H(Y_t+1|Y_t) (nats per second): {H_yy_test * len(y_events) / SIMULATION_TIME}")
    # print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per event): {H_yyx_test}")
    # print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx_test * len(y_events) / SIMULATION_TIME}")
    # print(f"Log Loss H(Y_t+1|Y_t): {log_loss_yy}")
    # print(f"Log Loss H(Y_t+1|Y_t,X_t): {log_loss_yyx}")

    run_multiple_estimation(
        target_events=torch.tensor(y_events),
        source_events=torch.tensor(x_events),
        configs=configs,
        n_runs=10,
        seed=seed
    )
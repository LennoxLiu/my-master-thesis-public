import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.integrate
from multiprocessing import Pool, cpu_count

# the intensity calculation logic from the simulation
def get_intensity_at_time(t, arrivals_self, arrivals_other,
                        mu, alpha_self, beta_self, alpha_other, beta_other, delay=0.02):
    intensity = mu
    for t_event in arrivals_self:
        if t_event < t:
            intensity += alpha_self * np.exp(-beta_self * (t - t_event))
    for t_event in arrivals_other:
        if t_event < t - delay:
            intensity += alpha_other * np.exp(-beta_other * (t - t_event - delay))
    return intensity

def simulate_hawkes(mu, alpha, beta, T):
    """
    Simulate a linear Hawkes process with exponential kernel.
    
    Args:
        mu (float): Baseline intensity.
        alpha (float): Excitation magnitude (must be < beta for stability).
        beta (float): Decay rate of excitation.
        T (float): Time horizon.
        
    Returns:
        list: Event times.
    """
    assert alpha < beta, "Stability condition violated: alpha must be < beta"
    
    t = 0
    events = []
    
    while t < T:
        current_intensity = mu + alpha * np.sum(np.exp(-beta * (t - np.array(events))))
        next_time = t + np.random.exponential(1 / current_intensity)
        
        if next_time > T:
            break
            
        events.append(next_time)
        t = next_time
    
    return torch.Tensor(events)


def simulate_mutually_exciting_hawkes(mu1, alpha11, beta11, alpha12, beta12,
                                     mu2, alpha22, beta22, alpha21, beta21,
                                     T_end, seed=None):
    """
    Simulates two mutually exciting Hawkes processes using the thinning algorithm.

    Args:
        mu1, mu2 (float): Base intensities for process 1 and 2.
        alpha11, beta11 (float): Self-excitation parameters for process 1.
        alpha12, beta12 (float): Excitation parameters from process 2 to process 1.
        alpha22, beta22 (float): Self-excitation parameters for process 2.
        alpha21, beta21 (float): Excitation parameters from process 1 to process 2.
        T_end (float): End time for the simulation.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (arrival_times_p1, arrival_times_p2) lists of arrival times.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    arrival_times_p1 = []
    arrival_times_p2 = []
    current_time = 0.0

    # Convert parameters to tensors for clarity if needed later for gradients,
    # but for simulation, floats are fine.
    # We'll keep them as floats for simplicity in the simulation loop.


    while current_time < T_end:
        # Estimate maximum intensity (an upper bound is crucial for thinning)
        # This is a simplification; a more robust approach would dynamically
        # track the maximum or use a fixed, sufficiently large upper bound.
        # For decaying kernels, the max intensity will generally be right after an event.
        # A quick heuristic: sum of base rates and max possible excitation from recent events.
        # This part is often the trickiest for a *tight* upper bound.
        # For simplicity, let's just take the current intensity as a lower bound for the max
        # and assume a reasonable cap.
        lambda_max_p1 = get_intensity_at_time(current_time, arrival_times_p1, arrival_times_p2,
                                            mu1, alpha11, beta11, alpha12, beta12) + alpha11 + alpha12 + mu1 # Add max possible jump
        lambda_max_p2 = get_intensity_at_time(current_time, arrival_times_p2, arrival_times_p1,
                                            mu2, alpha22, beta22, alpha21, beta21) + alpha22 + alpha21 + mu2 # Add max possible jump

        lambda_max_total = lambda_max_p1 + lambda_max_p2

        # If total intensity is very low, prevent division by zero or inf loop with tiny steps
        if lambda_max_total < 1e-9:
            # If no activity, jump to end or next event if mu > 0
            if mu1 > 0 or mu2 > 0:
                current_time += np.random.exponential(1.0 / (mu1 + mu2))
            else:
                current_time = T_end # No more events expected
            continue


        # Generate a candidate event time
        delta_t = np.random.exponential(1.0 / lambda_max_total)
        candidate_time = current_time + delta_t

        if candidate_time > T_end:
            break # Exceeds simulation time

        # Calculate actual intensities at the candidate time
        actual_lambda_p1 = get_intensity_at_time(candidate_time, arrival_times_p1, arrival_times_p2,
                                               mu1, alpha11, beta11, alpha12, beta12)
        actual_lambda_p2 = get_intensity_at_time(candidate_time, arrival_times_p2, arrival_times_p1,
                                               mu2, alpha22, beta22, alpha21, beta21)
        actual_lambda_total = actual_lambda_p1 + actual_lambda_p2

        # Thinning step: Accept or reject
        if np.random.rand() < (actual_lambda_total / lambda_max_total):
            # Accepted event - now decide which process generated it
            if np.random.rand() < (actual_lambda_p1 / actual_lambda_total):
                arrival_times_p1.append(candidate_time)
            else:
                arrival_times_p2.append(candidate_time)
        
        current_time = candidate_time

    return torch.tensor(arrival_times_p1), torch.tensor(arrival_times_p2)


# Simulate a one-step Hawkes process
# This function is not used in the main code but can be useful for testing.
# history_length in seconds
def simulate_hawkes_onestep(history = [0.001,0.003,0.005], mu=0.25, alpha=0.6, beta=0.8, history_length=1e-3*128, num_samples=1000):
    """
    Simulate a one-step Hawkes process with exponential kernel.
    
    Args:
        mu (float): Baseline intensity.
        alpha (float): Excitation magnitude (must be < beta for stability).
        beta (float): Decay rate of excitation.
        history_length (float): Length of the history in seconds.
        
    Returns:
        list: Event times.
    """
    assert alpha < beta, "Stability condition violated: alpha must be < beta"
    
    t = history[-1]
    while t < history_length and len(history) > 0:
        current_intensity = mu + alpha * np.sum(np.exp(-beta * (t - np.array(history))))
        next_time = t + np.random.exponential(1 / current_intensity)
        
        if next_time > history_length:
            break

        history.append(next_time)
        t = next_time

    print(f"Generated history of length {len(history)} with last time {history[-1]:.4f} seconds")
    t = history[-1]
    current_intensity = mu + alpha * np.sum(np.exp(-beta * (t - np.array(history))))
    targets = torch.Tensor(t + np.random.exponential(1 / current_intensity, size=(num_samples,)))
    histories = [torch.Tensor(history) for _ in range(len(targets))]
    return histories, targets



def _calculate_single_event_transfer_entropy(args):
    """
    Helper function to calculate transfer entropy for a single arrival_time.
    This function will be mapped to a multiprocessing pool.
    """
    arrival_time, target_process, source_process, params = args
    mu1, alpha11, beta11, alpha12, beta12, mu2, alpha22, beta22, alpha21, beta21 = params

    def cond_prob_yyx(t, target_proc_local, source_proc_local):
        intensity = get_intensity_at_time(t, target_proc_local, source_proc_local,
                                          mu1, alpha11, beta11, alpha12, beta12)

        t_last = np.max([0] + target_proc_local[target_proc_local < t].tolist()
                        + source_proc_local[source_proc_local < t].tolist())

        def survival_func(t_val):
            lambda_1 = get_intensity_at_time(t_val, target_proc_local, source_proc_local,
                                            mu1, alpha11, beta11, alpha12, beta12)
            lambda_2 = get_intensity_at_time(t_val, source_proc_local, target_proc_local,
                                            mu2, alpha22, beta22, alpha21, beta21)
            return lambda_1 + lambda_2

        survival_prob = np.exp(-scipy.integrate.quad(survival_func, t_last, t, limit=400)[0])
        return intensity * survival_prob

    def cond_prob_yy(t, target_proc_local, source_proc_local):
        intensity = get_intensity_at_time(t, target_proc_local, source_proc_local,
                                          mu1, alpha11, beta11, 0, 0)  # No excitation from source to target

        t_last = np.max([0] + target_proc_local[target_proc_local < t].tolist())

        def survival_func(t_val):
            lambda_1 = get_intensity_at_time(t_val, target_proc_local, source_proc_local,
                                            mu1, alpha11, beta11, 0, 0) # No excitation from source to target
            return lambda_1

        survival_prob = np.exp(-scipy.integrate.quad(survival_func, t_last, t, limit=400)[0])
        return intensity * survival_prob

    p_yyx = cond_prob_yyx(arrival_time, target_process, source_process)
    p_yy = cond_prob_yy(arrival_time, target_process, source_process)

    if p_yy > 0 and p_yyx > 0:  # Ensure p_yyx is also positive to avoid log(0)
        return np.log(p_yyx / p_yy)
    return 0.0 # Return 0 if p_yy is 0 or p_yyx is 0, as log(x/0) is undefined and log(0) is undefined

# This function is currently inaccurate
def numerical_hawkes_transfer_entropy_parallel(event_times, mu1, alpha11, beta11, alpha12, beta12,
                                                mu2, alpha22, beta22, alpha21, beta21):
    """
    Calculate the transfer entropy numerically for mutually exciting Hawkes processes using multiprocessing.
    Args:
        event_times (list): List of event times for the first process.
        mu1, mu2 (float): Base intensities for process 1 and 2.
        alpha11, beta11 (float): Self-excitation parameters for process 1.
        alpha12, beta12 (float): Excitation parameters from process 2 to process 1.
        alpha22, beta22 (float): Self-excitation parameters for process 2.
        alpha21, beta21 (float): Excitation parameters from process 1 to process 2.
    Returns:
        float: Estimated averaged transfer entropy for each inter-event time.
    """
    assert len(event_times) == 2, "Two processes are required for transfer entropy calculation."

    target_process = event_times[0].cpu().numpy()
    source_process = event_times[1].cpu().numpy()

    # Pack parameters for the helper function
    params = (mu1, alpha11, beta11, alpha12, beta12, mu2, alpha22, beta22, alpha21, beta21)

    # Prepare arguments for multiprocessing.Pool.map
    # Each item in `pool_args` will be passed as `args` to `_calculate_single_event_transfer_entropy`
    pool_args = [(arrival_time, target_process, source_process, params) for arrival_time in target_process]

    # Use a multiprocessing Pool to parallelize the calculations
    # `cpu_count()` gets the number of CPU cores available
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_calculate_single_event_transfer_entropy, pool_args)

    transfer_entropy = sum(results)
    if len(target_process) > 0:
        transfer_entropy /= len(target_process)
    else:
        return 0.0 # Handle case where target_process is empty

    return transfer_entropy

if __name__ == "__main__":
    # Example usage
    # mu, alpha, beta, T = 0.75, 0.6, 0.8, 60
    # events = simulate_hawkes(mu, alpha, beta, T)
    # print(events[:10])

    # print(f"Simulated {len(events)} events.")
    # print(f"Analytical entropy rate: {hawkes_entropy_rate(mu, alpha, beta):.4f} bits/unit time")

    # # Plot intensity and events
    # def plot_hawkes(events, mu, alpha, beta, T):
    #     t_grid = np.linspace(0, T, 1000)
    #     intensity = np.zeros_like(t_grid)
        
    #     for i, t in enumerate(t_grid):
    #         past_events = events[events < t]
    #         intensity[i] = mu + alpha * np.sum(np.exp(-beta * (t - past_events)))
        
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(t_grid, intensity, label='Intensity λ(t)')
    #     plt.vlines(events, 0, 0.5, color='red', alpha=0.3, label='Events')
    #     plt.xlabel('Time')
    #     plt.ylabel('Intensity')
    #     plt.legend()
    #     plt.show()

    # plot_hawkes(events, mu, alpha, beta, T)
    # # Plot histogram of inter-event times
    # inter_event_times = np.diff(events)
    # plt.figure(figsize=(8, 4))
    # plt.hist(np.log10(inter_event_times), bins=100, color='skyblue', edgecolor='black')
    # plt.xlabel('Log10 Inter-event time')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Inter-event Times')
    # # plt.xscale('log')
    # plt.show()

    # # Plot results of simulate_hawkes_onestep
    # histories, targets = simulate_hawkes_onestep([0.001,0.002,0.003,0.004],num_samples=10000)
    # plt.figure(figsize=(10, 4))
    # plt.hist(np.log10(targets.numpy()), bins=150, alpha=0.5, color='blue', label='Targets')
    # plt.xlabel('Log Time')
    # plt.ylabel('Counts')
    # plt.title('Simulated Hawkes One-Step Histories and Targets')
    # plt.show()


    # Define parameters (ensure alpha < beta * (1 - sum of other alphas for stability for each process, for mutual its more complex)
    # For stability of a mutually exciting Hawkes process, you typically need the spectral radius of the matrix
    # of "marks" (alpha/beta ratios) to be less than 1. This is a general guideline.
    # Here, let's just pick some values that should produce events.
    
    # Process 1 parameters
    mu1 = 0.1
    alpha11 = 0  # Self-excitation P1 -> P1
    beta11 = 0
    alpha12 = 1.5  # Excitation P2 -> P1
    beta12 = 0.8  # Decay rate of excitation from P2 to P1

    # Process 2 parameters
    mu2 = 10
    alpha22 = 0.1  # Self-excitation P2 -> P2
    beta22 = 0.8
    alpha21 = 0.1  # Excitation P1 -> P2 (stronger influence from P1 to P2)
    beta21 = 0.8

    T_end = 60  # Simulation duration

    print("Simulating mutually exciting Hawkes processes...")
    arrival_times_p1, arrival_times_p2 = simulate_mutually_exciting_hawkes(
        mu1, alpha11, beta11, alpha12, beta12,
        mu2, alpha22, beta22, alpha21, beta21,
        T_end, seed=31
    )

    print(f"Process 1 generated {len(arrival_times_p1)} events.")
    print(f"Process 2 generated {len(arrival_times_p2)} events.")

    # --- Visualize Intensity ---
    num_points = 500 # Number of points to sample for intensity visualization
    time_points = np.linspace(0, T_end, num_points)

    intensities_p1 = []
    intensities_p2 = []

    print("\nCalculating intensities for visualization...")
    for t in time_points:
        intensities_p1.append(get_intensity_at_time(t, arrival_times_p1, arrival_times_p2,
                                                    mu1, alpha11, beta11, alpha12, beta12))
        intensities_p2.append(get_intensity_at_time(t, arrival_times_p2, arrival_times_p1,
                                                    mu2, alpha22, beta22, alpha21, beta21))


    # You can also plot both intensities on the same graph if they have similar scales
    plt.figure(figsize=(14, 5))
    plt.plot(time_points, intensities_p1, label='Intensity P1', color='blue')
    plt.plot(time_points, intensities_p2, label='Intensity P2', color='red')
    plt.eventplot(arrival_times_p1, lineoffsets=0, linelengths=0.5, color='blue', alpha=0.6)
    plt.eventplot(arrival_times_p2, lineoffsets=0, linelengths=0.5, color='red', alpha=0.6)
    plt.xlabel("Time")
    plt.ylabel("Intensity")
    plt.title("Mutually Exciting Hawkes Process Intensities")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, T_end)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()


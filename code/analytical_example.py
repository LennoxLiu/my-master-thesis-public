from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import time
import numpy as np
from juliacall import Pkg as jl_pkg
from juliacall import Main as jl
import torch
print("Julia runtime started via juliacall.")
project_path = "D:/Code/CoTETE.jl/"
print(f"Activating project at: {project_path}")
jl_pkg.activate(project_path)
print("Instantiating project dependencies...")
jl_pkg.instantiate()
print("Loading CoTETE module...")
jl.seval("using CoTETE")
CoTETE = jl.CoTETE
print("CoTETE module loaded successfully.")
from entropy_tpp import TE_estimation_tpp, run_multiple_estimation

class SpikingNeuronModel:
    """
    Implementation of the spiking neuron model from Section V.A using thinning algorithm
    Transfer entropy in continuous time, with applications to jump and neural spiking processes
    """
    
    def __init__(self, lambda_y=1.0, a=0.5, tau=1.0, tau_r=1.0):
        """
        Parameters:
        -----------
        lambda_y : float
            Source spike rate (outside refractory period)
        a : float
            Probability of target spiking within tau seconds after source spike
        tau : float
            Duration of elevated rate period after source spike
        tau_r : float
            Refractory period duration for both neurons
        """
        self.lambda_y = lambda_y
        self.a = a
        self.tau = tau
        self.tau_r = tau_r
        
        # Calculate elevated spike rate
        self.lambda_e_xy = -np.log(1 - a) / tau
    
    def get_intensity_y(self, t, t_y=-np.inf):
        """
        Get intensity function for source neuron y at time t
        
        """
        result = 0.0
        # print("t={}, t_y={}, self.tau_r={}".format(t, t_y, self.tau_r))
        if t > t_y + self.tau_r:
            result = self.lambda_y
        
        return result
    
    def get_intensity_x(self, t, t_x=-np.inf, t_y=0):
        """
        Get intensity function for target neuron x at time t
        
        """
        result = 0.0
        # Check refractory period for x

        if t_y < t and t <= t_y + self.tau and t_x <= t_y and t > t_x + self.tau_r:
                result = self.lambda_e_xy
        
        return result
    
    def simulate_thinning(self, T=100.0, seed=None):
        """
        Simulate spike trains using Ogata's thinning algorithm
        
        The thinning algorithm (also called rejection sampling):
        1. Find upper bound lambda_max for intensity function
        2. Generate candidate events from homogeneous Poisson(lambda_max)
        3. Accept each candidate with probability lambda(t)/lambda_max
        
        Parameters:
        -----------
        T : float
            Total simulation time
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        spike_times_source : array
            Spike times for source neuron
        spike_times_target : array
            Spike times for target neuron
        """
        if seed is not None:
            np.random.seed(seed)
        
        spike_times_source = []
        spike_times_target = []
        
        # Simulate source y first (independent process)
        t = 0
        while t < T:
            # Upper bound for intensity (lambda_y when not in refractory)
            lambda_max_y = self.lambda_y
            
            t_y = t
            # Generate inter-event time from homogeneous Poisson
            t += np.random.exponential(1.0 / lambda_max_y)
            
            if t >= T:
                break
            
            # Thinning: accept with probability lambda(t)/lambda_max
            intensity = self.get_intensity_y(t, t_y)
            if np.random.rand() < intensity / lambda_max_y:
                spike_times_source.append(t)
        
        spike_times_source = np.array(spike_times_source)
        # Simulate target x (depends on y)
        t = 0
        while t < T:
            # Upper bound for intensity
            lambda_max_x = max(self.lambda_e_xy, 1e-10)  # Avoid division by zero
            
            t_x = t
            # Generate inter-event time
            t += np.random.exponential(1.0 / lambda_max_x)
            
            if t >= T:
                break
            
            # find largest spike_times_source value strictly less than t
            t_y = spike_times_source[spike_times_source < t]
            if len(t_y) == 0:
                t_y = 0
            else:
                t_y = t_y[-1]

            # Thinning: accept with probability lambda(t)/lambda_max
            intensity = self.get_intensity_x(t, t_x, t_y=t_y)
            if intensity > 0 and np.random.rand() < intensity / lambda_max_x:
                spike_times_target.append(t)
        
        return spike_times_source, np.array(spike_times_target)
    

    
    def calculate_transfer_entropy_rate_analytical(self):
        """
        Calculate analytical transfer entropy rate (Equation 38)
        Valid to O(lambda_y)
        
        Returns:
        --------
        TE_rate : float
            Transfer entropy rate in nats/sec
        """
        if self.a <= 0 or self.a >= 1:
            return np.inf
        
        TE_rate = self.a * self.lambda_y * np.log(-np.log(1-self.a) / (self.a * self.lambda_y * self.tau) )
        
        return TE_rate
    
    def calculate_coarse_grained_rate_lambda_x(self, t, t_x1=0):
        """
        Calculate coarse-grained spike rate lambda_x(t, t_x1=0) (Equation 37)
        Valid for O(lambda_y)
        
        Parameters:
        -----------
        t : float or array
            Time since last spike in x
        t_x1 : float
            Time of last spike (set to 0 for convenience)
            
        Returns:
        --------
        lambda_x : float or array
            Coarse-grained spike rate
        """
        t = np.asarray(t)
        lambda_x = np.zeros_like(t, dtype=float)
        
        # Region 1: 0 <= t < tau_r (refractory)
        mask1 = (t >= 0) & (t < self.tau_r)
        lambda_x[mask1] = 0
        
        # Region 2: tau_r <= t < tau_r + tau (growing rate)
        mask2 = (t >= self.tau_r) & (t < self.tau_r + self.tau)
        lambda_x[mask2] = (1 - (1 - self.a)**((t[mask2] - self.tau_r) / self.tau)) * self.lambda_y
        
        # Region 3: t >= tau_r + tau (constant rate)
        mask3 = t >= self.tau_r + self.tau
        lambda_x[mask3] = self.a * self.lambda_y
        
        return lambda_x


def plot_transfer_entropy_vs_parameters():
    """
    Reproduce Figure 1 from the paper: TE rate vs coupling strength for different lambda_y
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parameters
    tau = 1.0
    tau_r = 1.0
    a_values = np.linspace(0.05, 0.95, 100)
    lambda_y_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    # Plot 1: Normalized TE rate
    for lambda_y in lambda_y_values:
        te_normalized = []
        for a in a_values:
            model = SpikingNeuronModel(lambda_y=lambda_y, a=a, tau=tau, tau_r=tau_r)
            te_rate = model.calculate_transfer_entropy_rate_analytical()
            # Normalize by limiting mean target spike rate
            te_normalized.append(te_rate / (a*lambda_y))
        
        ax1.plot(a_values, te_normalized, label=f'λ_y = {lambda_y}', linewidth=2)
    
    ax1.set_xlabel('Coupling strength (a)', fontsize=12)
    ax1.set_ylabel(' Ṫ_y→x (normalized)', fontsize=12)
    ax1.set_title('Normalized Transfer Entropy Rate vs Coupling Strength', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim([0, 14])

    # Plot 2: Lambda_x over time
    model = SpikingNeuronModel(lambda_y=0.5, a=0.7, tau=tau, tau_r=tau_r)
    t_values = np.linspace(0, 3, 300)
    lambda_x = model.calculate_coarse_grained_rate_lambda_x(t_values)
    
    ax2.plot(t_values, lambda_x, linewidth=2, color='purple')
    ax2.axvline(tau_r, color='red', linestyle='--', alpha=0.5, label='τ_r')
    ax2.axvline(tau_r + tau, color='orange', linestyle='--', alpha=0.5, label='τ_r + τ')
    ax2.axhline(model.a * model.lambda_y, color='green', linestyle='--', alpha=0.5, label='a·λ_y')
    
    ax2.set_xlabel('Time since last x spike (t)', fontsize=12)
    ax2.set_ylabel('λ_x(t)', fontsize=12)
    ax2.set_title('Coarse-Grained Spike Rate λ_x(t)', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spike_trains(spike_times_source, spike_times_target, T_max=None):
    """
    Visualize spike trains for source and target neurons
    """
    if T_max is None:
        T_max = max(spike_times_source[-1] if len(spike_times_source) > 0 else 0,
                    spike_times_target[-1] if len(spike_times_target) > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    # Source spikes
    ax1.eventplot(spike_times_source, colors='blue', linewidths=2)
    ax1.set_ylabel('Source (y)', fontsize=12)
    ax1.set_ylim([0.5, 1.5])
    ax1.set_yticks([])
    ax1.set_title(f'Spike Trains (Total: y={len(spike_times_source)}, x={len(spike_times_target)})', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Target spikes
    ax2.eventplot(spike_times_target, colors='green', linewidths=2)
    ax2.set_ylabel('Target (x)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylim([0.5, 1.5])
    ax2.set_yticks([])
    ax2.set_xlim([0, T_max])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_intensity_functions(model, spike_times_source, spike_times_target, T_max=50):
    """
    Plot the time-varying intensity functions alongside spike trains
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Time grid for intensity evaluation
    t_grid = np.linspace(0, T_max, 10000)
    
    # Calculate intensity functions
    intensity_y = np.zeros_like(t_grid)
    intensity_x = np.zeros_like(t_grid)
    
    for i, t in enumerate(t_grid):
        sy = spike_times_source[spike_times_source < t]
        if len(sy) == 0:
            sy = 0
        else:
            sy = sy[-1]
        sx = spike_times_target[spike_times_target < t]
        if len(sx) == 0:
            sx = 0
        else:
            sx = sx[-1]
        intensity_y[i] = model.get_intensity_y(t, sy)
        intensity_x[i] = model.get_intensity_x(t, sx, sy)
    
    # Plot source intensity
    axes[0].plot(t_grid, intensity_y, 'b-', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('λ_y(t)', fontsize=11)
    axes[0].set_title('Source Intensity Function', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, model.lambda_y * 1.2])
    
    # Plot source spikes
    axes[1].eventplot([spike_times_source[spike_times_source < T_max]], colors='blue', linewidths=2)
    axes[1].set_ylabel('y spikes', fontsize=11)
    axes[1].set_ylim([0.5, 1.5])
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Plot target intensity
    axes[2].plot(t_grid, intensity_x, 'g-', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('λ_x|y(t)', fontsize=11)
    axes[2].set_title('Target Intensity Function (given y)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.1, model.lambda_e_xy * 1.2])
    
    # Plot target spikes
    axes[3].eventplot([spike_times_target[spike_times_target < T_max]], colors='green', linewidths=2)
    axes[3].set_ylabel('x spikes', fontsize=11)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].set_ylim([0.5, 1.5])
    axes[3].set_yticks([])
    axes[3].set_xlim([0, T_max])
    axes[3].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig



if __name__ == "__main__":
    """
    Main simulation and analysis using thinning algorithm
    """
    print("=" * 70)
    print("Transfer Entropy: Spiking Neuron Model (Thinning Algorithm)")
    print("=" * 70)
    
    # Model parameters
    lambda_y = 0.01 # analytical result has residual O(lambda_y^2)
    a = 0.8 # 0 < a < 1
    tau = 100.0
    tau_r = 100.0 # tau_r >= tau
    T = 100000
    
    print(f"\nModel Parameters:")
    print(f"  Source rate (λ_y): {lambda_y}")
    print(f"  Coupling strength (a): {a}")
    print(f"  Window duration (τ): {tau}")
    print(f"  Refractory period (τ_r): {tau_r}")
    print(f"  Simulation time: {T}")
    
    # Create model
    model = SpikingNeuronModel(lambda_y=lambda_y, a=a, tau=tau, tau_r=tau_r)
    
    # Calculate analytical transfer entropy rate
    te_rate = model.calculate_transfer_entropy_rate_analytical()
    print(f"\n{'Analytical Transfer Entropy Rate':^70}")
    print(f"{'-' * 70}")
    print(f"  Ṫ_y→x = {te_rate:.6f} nats/sec")
    print(f"  Normalized (Ṫ_y→x / (λ_y)) = {te_rate / (lambda_y):.6f}")
    print(f"  Formula: a·λ_y·ln[-ln(1-a) / (a·λ_y·τ)]")
    
    # Simulate spike trains using thinning algorithm
    print(f"\n{'Simulating with Thinning Algorithm':^70}")
    print(f"{'-' * 70}")
    spike_times_source, spike_times_target = model.simulate_thinning(T=T, seed=42)
    
    print(f"  Generated {len(spike_times_source)} source spikes")
    print(f"  Generated {len(spike_times_target)} target spikes")
    print(f"  Empirical source rate: {len(spike_times_source) / T:.4f} spikes/sec")
    print(f"  Expected source rate: ~{lambda_y:.4f} spikes/sec")
    print(f"  Empirical target rate: {len(spike_times_target) / T:.4f} spikes/sec")
    print(f"  Expected target rate: ~{a * lambda_y:.4f} spikes/sec")
    
    # # Show inter-spike intervals
    if len(spike_times_target) > 1:
        isi_x = np.diff(spike_times_target)
        print(f"  Target ISI: mean={np.mean(isi_x):.3f}, std={np.std(isi_x):.3f}")
    if len(spike_times_source) > 1:
        isi_y = np.diff(spike_times_source)
        print(f"  Source ISI: mean={np.mean(isi_y):.3f}, std={np.std(isi_y):.3f}")
    
    # # Generate plots
    # print(f"\n{'Generating plots...':^70}")
    
    # Plot 1: TE rate vs parameters (Figure 1)
    fig1 = plot_transfer_entropy_vs_parameters()
    plt.figure(fig1.number)
    plt.show()
    plt.close(fig1)
    plt.savefig('./results/transfer_entropy_vs_parameters.png', dpi=150, bbox_inches='tight')
    print("  Saved: transfer_entropy_vs_parameters.png")
    
    # # Plot 2: Spike trains
    # fig2 = plot_spike_trains(spike_times_source, spike_times_target)
    # plt.figure(fig2.number)
    # plt.show()
    # plt.close(fig2)
    # plt.savefig('./results/spike_trains.png', dpi=150, bbox_inches='tight')
    # print("  Saved: spike_trains.png")
    
    # # Plot 3: Intensity functions
    # fig3 = plot_intensity_functions(model, spike_times_source, spike_times_target, T_max=50)
    # plt.figure(fig3.number)
    # plt.savefig('./results/intensity_functions.png', dpi=150, bbox_inches='tight')
    # print("  Saved: intensity_functions.png")
    
    # plt.show()
    
    # print(f"\n{'Simulation complete!':^70}")
    # print("=" * 70)

    seed=42
    for lx in [1,2,4]:
        for ly in [1,2,4]:
            start_time = time.time()
            params = CoTETE.CoTETEParameters(l_x = lx, l_y = ly) # l_x target length, l_y source length
            result = CoTETE.estimate_TE_from_event_times(params, jl.Array(spike_times_target), jl.Array(spike_times_source))
            end_time = time.time()
            duration = end_time - start_time
            print(f"CoTETE TE Estimation for l_x={lx}, l_y={ly}: Estimated Transfer Entropy: {result} nats per second, Completed in {duration/60:.2f} minutes")


    # ================ TE tpp estimation using neural network model ================
    torch.manual_seed(seed)
    np.random.seed(seed)
    time_series_length = T   # in seconds, Length of the time series
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
        "plot_histograms": False,  # Whether to plot the conditional histograms
        "plot_pp": False,            # Whether to plot the probability - probability plots
        "history_length": 8,             # in number of bins, Length of the history to use for the model
        "num_mc_samples": 20480,        # Number of Monte Carlo samples for entropy estimation
    }
    spike_times_target = torch.tensor(spike_times_target, dtype=torch.float32)
    spike_times_source = torch.tensor(spike_times_source, dtype=torch.float32)
    # start_time = time.time()
    # (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
    #         event_time=[spike_times_target, spike_times_source], 
    #         configs=configs, 
    #         seed=seed
    # )
    # end_time = time.time()
    # duration = end_time - start_time
    # print(f"\n--- TE Estimation Completed in {duration/60:.2f} minutes ---")
    # print(f"Estimated Transfer Entropy (nats per second): {TE_test:.5f}")
    # print(f"Estimated H(Y_t+1|Y_t) (nats per second): {H_yy_test:.5f}")
    # print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx_test:.5f}")
    # print(f"Log Loss H(Y_t+1|Y_t): {log_loss_yy:.5f}")
    # print(f"Log Loss H(Y_t+1|Y_t,X_t): {log_loss_yyx:.5f}")

    # run_multiple_estimation(
    #     target_events=spike_times_target,
    #     source_events=spike_times_source,
    #     configs=configs,
    #     n_runs=10,
    #     seed=seed
    # )
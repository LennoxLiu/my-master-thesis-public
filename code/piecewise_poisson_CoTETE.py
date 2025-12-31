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

from entropy_tpp import TE_estimation_tpp, save_dict_indented
from piecewise_poisson import simulate_processes, compute_reference

# # 5. Your original code
# params = CoTETE.CoTETEParameters(l_x = 1, l_y = 1)

# target = 1e3*np.random.rand(1000)
# target = np.sort(target)
# source = 1e3*np.random.rand(1000)
# source = np.sort(source)

# print("Running CoTETE.estimate_TE_from_event_times...")
# result = CoTETE.estimate_TE_from_event_times(params, jl.Array(target), jl.Array(source))
# print(f"Result: {result}")

if __name__ == "__main__":
    # Define simulation parameters
    SIMULATION_TIME = 15*60  # seconds
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
            "context_size": 4,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 4,        # 16 Number of components for a mixture model
            "hidden_sizes": [32, 32],     # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "model_config_yyx": {
            "model_name": "LogNormMix_yyx",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 4,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 8,  # 16 Number of components fo r a mixture model
            "hidden_sizes": [32, 32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "train_config_yy": {
            "L2_weight": 1e-3,          # L2 regularization parameter
            # "L_entropy_weight": 1e-1,      # Weight for the entropy regularization term
            # "L_sep_weight": 1e-5,               # Weight for the separation regularization term
            "L_scale_weight": 1e-2,             # Weight for the scale regularization term
            # "L_mean_match_weight": 1,        # Weight for the mean matching regularization term
            "learning_rate": 1e-3,           # Learning rate for Adam optimizer
            "max_epochs": 500,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 20,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "train_config_yyx": {
            "L2_weight": 1e-3,          # L2 regularization parameter
            # "L_entropy_weight": 1e-1,      # Weight for the entropy regularization term
            # "L_sep_weight": 1e-5,               # Weight for the separation regularization term
            "L_scale_weight": 1e-2,             # Weight for the scale regularization term
            # "L_mean_match_weight": 1,        # Weight for the mean matching regularization term
            "learning_rate": 1e-3,           # Learning rate for Adam optimizer
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
    params = CoTETE.CoTETEParameters(l_x = 1, l_y = 2)
    # - `l_x::Integer`: The number of intervals in the target process to use in the history embeddings.
    # Corresponds to ``l_X`` in [^1].
    # - `l_y::Integer`: The number of intervals in the source process to use in the history embeddings.
    # Corresponds to ``l_Y`` in [^1].
    CoTETE_results = []
    for i in range(10):
        result = CoTETE.estimate_TE_from_event_times(params, jl.Array(np.array(y_events)), jl.Array(np.array(x_events)))
        CoTETE_results.append(result)
        print(f"CoTETE run {i+1}/10: Estimated Transfer Entropy: {result:.5f} nats per second")
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- CoTETE TE Estimation Completed in {duration/60:.2f} minutes ---")
    average_TE = sum(CoTETE_results) / len(CoTETE_results)
    print(f"\n--- CoTETE Average Estimated Transfer Entropy over 10 runs: {average_TE:.5f} nats per second ---")

    # start_time = time.time()
    # (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
    #         event_time=[torch.tensor(y_events), torch.tensor(x_events)], 
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
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from exp_data_loader import load_grouped_data, get_list_by_length_criteria
from piecewise_lognormal import simulate_processes, compute_reference
import optuna
from entropy_tpp import run_multiple_estimation
from entropy_tpp import save_dict_indented

# Define simulation parameters
seed=76  # For reproducibility

# Load experimental data
bc_data, pom_data = load_grouped_data()
print(f"Loaded {len(bc_data)} BC neurons and {len(pom_data)} POm neurons.")
time_series_length = 1*60   # in seconds, Length of the time series, in the same unit of data

# Data preprocessing, remove neurons with too few spikes
min_spikes = time_series_length * 30  # Minimum number of spikes required 30 Hz
bc_data = [neuron for neuron in bc_data if len(neuron) >= min_spikes]
pom_data = [neuron for neuron in pom_data if len(neuron) >= min_spikes]
print(f"After filtering, {len(bc_data)} BC neurons and {len(pom_data)} POm neurons remain.")


torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

target = torch.tensor(get_list_by_length_criteria(pom_data, 'longest'), dtype=torch.float32)
source = torch.tensor(get_list_by_length_criteria(bc_data, 'longest'), dtype=torch.float32)

configs = {
            "model_config_yyx": {
                "model_name": "yyx",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
                "context_size": 16,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
                "num_mix_components": 4,  # 16 Number of components for a mixture model
                "hidden_sizes": [4, 32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": "GELU",
            },
            "train_config_yyx": {
                "L2_weight": 0.000827406850573813,          # L2 regularization parameter
                "L_entropy_weight": 0.0001264625733948674,      # Weight for the entropy regularization term
                "L_sep_weight": 2.9247292664457833e-05,               # Weight for the separation regularization term
                "L_scale_weight":  3.3454456195454425e-09,             # Weight for the scale regularization term
                "learning_rate": 5e-4,           # Learning rate for Adam optimizer
                "max_epochs": 1000,              # For how many epochs to train
                "display_step": 5,               # Display training statistics after every display_step
                "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
            },
            "model_config_yy": {
                "model_name": "yy",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
                "context_size": 16,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
                "num_mix_components": 8,  # 16 Number of components for a mixture model
                "hidden_sizes": [32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": "GELU",
            },
            "train_config_yy": {
                "L2_weight": 0.0001907358691282098,          # L2 regularization parameter
                "L_entropy_weight": 0.00021180598933158608,      # Weight for the entropy regularization term
                "L_sep_weight": 1.6394325044339145e-07,               # Weight for the separation regularization term
                "L_scale_weight": 6.378480299432731e-06,             # Weight for the scale regularization term
                "learning_rate": 5e-4,           # Learning rate for Adam optimizer
                "max_epochs": 1000,              # For how many epochs to train
                "display_step": 5,               # Display training statistics after every display_step
                "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
            },
            "data_prep_config":{
                "batch_size": 128,          # Number of sequences in a batch
                "shuffle": False,                 # Whether to shuffle the time series before splitting into train/val/test
                "total_time": time_series_length,              # in second, Total time of the sequences, truncated to this length for training
                "verbose": False
            },
            "device": device,
            "verbose": False,  # Whether to print the training statistics
            "plot_histograms": False,  # Whether to plot the conditional histograms
            "plot_pp": False,  # Whether to plot the P-P plot
            "history_length": 32,             # in number of bins, Length of the history to use for the model
        }

# Save the config for reference
save_dict_indented(configs, f"./results/config_{seed}.txt")

run_multiple_estimation(target, source, configs=configs, n_runs=20, seed=seed)
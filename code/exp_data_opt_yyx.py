from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from exp_data_loader import load_grouped_data, get_list_by_length_criteria
from piecewise_lognormal import simulate_processes, compute_reference
import optuna
from entropy_tpp import CondH_estimation_yyx



def create_objective(arrival_times_target_list, arrival_times_source_list,
                     time_series_length, device, seed):
    """
    This outer function creates and returns the actual objective function.
    It takes the data as an argument.
    """
    
    def objective(trial):
        # 1. Suggest hyperparameters

        n_layers_yyx = trial.suggest_int("n_layers_yyx", 1, 2) # From 1 to 5 hidden layers
        hidden_sizes_yyx = []
        for i in range(n_layers_yyx):
            # Suggest the size for each hidden layer dynamically
            layer_size = 2** trial.suggest_int(f"hidden_size_yyx_l{i}", 2, 5)
            hidden_sizes_yyx.append(layer_size)

        configs = {
            "model_config_yyx": {
                "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
                "context_size": 2** trial.suggest_int("context_size_yyx", 0, 4),  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
                "num_mix_components": 2** trial.suggest_int("num_mix_components_yyx", 1, 3),  # 16 Number of components for a mixture model
                "hidden_sizes": hidden_sizes_yyx,       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "context_extractor": trial.suggest_categorical("context_extractor_yyx", ["gru", "lstm", "mlp"]), # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": trial.suggest_categorical("activation_func_yyx", ["Tanh", "ReLU", "GELU"]),
            },
            "train_config_yyx": {
                "L2_weight": trial.suggest_float("L2_weight_yyx", 1e-5, 2e-3, log=True),          # L2 regularization parameter
                "L_entropy_weight": trial.suggest_float("L_entropy_weight_yyx", 1e-10, 1e-3, log=True),      # Weight for the entropy regularization term
                "L_sep_weight": trial.suggest_float("L_sep_weight_yyx", 1e-10, 1e-3, log=True),               # Weight for the separation regularization term
                "L_scale_weight": trial.suggest_float("L_scale_weight_yyx", 1e-10, 1e-3, log=True),             # Weight for the scale regularization term
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
            "history_length": 32,             # in number of bins, Length of the history to use for the model
        }
        
        # 2. Run your TE estimation with the suggested hyperparameters
        # You'll need to adapt this part to call the relevant functions in your entropy_tpp.py
        # For instance, if you have a main function that takes these as arguments:
        # te_train, te_val, te_test = calculate_te(param1, param2)
        
        H_yyx_tests_sec = []
        print("Number of neurons in target process:", len(arrival_times_target_list))
        print("Number of neurons in source process:", len(arrival_times_source_list))

        # filtered_source = [torch.tensor(get_list_by_length_criteria(arrival_times_source_list, 'shortest'), dtype=torch.float32),
        #                    torch.tensor(get_list_by_length_criteria(arrival_times_source_list, 'middle'), dtype=torch.float32),
        #                    torch.tensor(get_list_by_length_criteria(arrival_times_source_list, 'longest'), dtype=torch.float32)]
        filtered_source = [torch.tensor(get_list_by_length_criteria(arrival_times_source_list, 'longest'), dtype=torch.float32)]
        # filtered_target = [torch.tensor(get_list_by_length_criteria(arrival_times_target_list, 'shortest'), dtype=torch.float32),
        #                    torch.tensor(get_list_by_length_criteria(arrival_times_target_list, 'middle'), dtype=torch.float32),
        #                    torch.tensor(get_list_by_length_criteria(arrival_times_target_list, 'longest'), dtype=torch.float32)]
        filtered_target = [torch.tensor(get_list_by_length_criteria(arrival_times_target_list, 'longest'), dtype=torch.float32)]

        log_yyx_losses = []
        i=0
        for arrival_times_target in filtered_target:
            for arrival_times_source in filtered_source:
                len_target = len(arrival_times_target)
                h_yyx, log_loss_yyx = CondH_estimation_yyx(
                    event_time=[arrival_times_target, arrival_times_source],
                    configs=deepcopy(configs),
                    seed=seed + i + 1
                )
                log_yyx_losses.append(log_loss_yyx)
                
                if  h_yyx == float('nan'):
                    print(f"Error during TE estimation for run {i+1}. Skipping this run.\n")
                    return None, None
                
                h_yyx_sec = h_yyx * len_target / time_series_length
            
                print(f'Conditional entropy h_yyx for run {i+1}: {h_yyx:.5f} nats/event, {h_yyx_sec:.5f} nats/sec')
                print(f'Log loss for model yy: {log_loss_yyx:.5f}')

                trial.set_user_attr(f"h_yyx_test_sec_run_{i}", h_yyx_sec)
                trial.set_user_attr(f"log_loss_yyx_run_{i}", log_loss_yyx)
                
                H_yyx_tests_sec.append(h_yyx_sec)
                i+=1
        
        # 3. Return the metric to optimize
        return  np.mean(log_yyx_losses)

    return objective

# Define simulation parameters
seed=74  # For reproducibility

# Load experimental data
bc_data, pom_data = load_grouped_data()
print(f"Loaded {len(bc_data)} BC neurons and {len(pom_data)} POm neurons.")
time_series_length = 15*60   # in seconds, Length of the time series, in the same unit of data

# Data preprocessing, remove neurons with too few spikes
min_spikes = time_series_length * 30  # Minimum number of spikes required 30 Hz
bc_data = [neuron for neuron in bc_data if len(neuron) >= min_spikes]
pom_data = [neuron for neuron in pom_data if len(neuron) >= min_spikes]
print(f"After filtering, {len(bc_data)} BC neurons and {len(pom_data)} POm neurons remain.")


torch.manual_seed(seed)
np.random.seed(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# I am looking at BC -> POm interactions here
objective_t = create_objective(pom_data, bc_data,
                     time_series_length, device, seed)

# Assuming 'objective' function is defined as above
# ,load_if_exists=True to continue from an existing study
study = optuna.create_study(directions=["minimize"], storage="sqlite:///db.sqlite3"
                            ,load_if_exists=True, study_name=f"exp_data_Hyyx_{seed:02d}") # Set direction to 'maximize' for TE,  

study.optimize(objective_t, n_trials=None) # Run for unlimited trials

print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")
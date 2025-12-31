from contextlib import redirect_stdout
from copy import deepcopy
import time
import numpy as np
import matplotlib.pyplot as plt
import optuna
from entropy_tpp import CondH_estimation_yy, TE_estimation_tpp, run_multiple_estimation, save_dict_indented
import torch
from piecewise_poisson import simulate_processes, compute_reference

def create_objective(arrival_times_target_list, arrival_times_source_list,
                     time_series_length, device, seed):
    """
    This outer function creates and returns the actual objective function.
    It takes the data as an argument.
    """
    
    def objective(trial):
        # 1. Suggest hyperparameters

        n_layers_yy = trial.suggest_int("n_layers_yy", 1, 3) # From 1 to 5 hidden layers
        hidden_sizes_yy = []
        for i in range(n_layers_yy):
            # Suggest the size for each hidden layer dynamically
            layer_size = 2** trial.suggest_int(f"hidden_size_yy_l{i}", 2, 6)
            hidden_sizes_yy.append(layer_size)

        configs = {
            "model_config_yy": {
                "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
                "context_size": 2** trial.suggest_int("context_size_yy", 0, 6),  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
                "num_mix_components": 2** trial.suggest_int("num_mix_components_yy", 1, 7),  # 16 Number of components for a mixture model
                "hidden_sizes": hidden_sizes_yy,       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "context_extractor": trial.suggest_categorical("context_extractor_yy", ["gru", "lstm", "mlp"]), # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": trial.suggest_categorical("activation_func_yy", ["Tanh", "ReLU", "GELU"]),
            },
            "train_config_yy": {
                "L2_weight": trial.suggest_float("L2_weight_yy", 1e-10, 1e-3, log=True),          # L2 regularization parameter
                # "L_entropy_weight": trial.suggest_float("L_entropy_weight_yy", 1e-10, 1e-3, log=True),      # Weight for the entropy regularization term
                # "L_sep_weight": trial.suggest_float("L_sep_weight_yy", 1e-10, 1e-3, log=True),               # Weight for the separation regularization term
                "L_scale_weight": trial.suggest_float("L_scale_weight_yy", 1e-10, 1e-3, log=True),             # Weight for the scale regularization term
                "learning_rate": trial.suggest_float("learning_rate_yy", 1e-5, 1e-2, log=True),           # Learning rate for Adam optimizer
                "max_epochs": 500,              # For how many epochs to train
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
            "plot_histograms": False,  # Whether to plot the conditional histograms
            "history_length": 16,             # in number of bins, Length of the history to use for the model
            "plot_pp": False,  # Whether to plot the PP plots
        }
        
        # 2. Run your TE estimation with the suggested hyperparameters
        # You'll need to adapt this part to call the relevant functions in your entropy_tpp.py
        # For instance, if you have a main function that takes these as arguments:
        # te_train, te_val, te_test = calculate_te(param1, param2)
        
        H_yy_tests_sec = []
        log_yy_losses = []
        for i in range(len(arrival_times_target_list)):
            arrival_times_target = arrival_times_target_list[i]
            arrival_times_source = arrival_times_source_list[i]

            print("Number of events in target process:", len(arrival_times_target))
            print("Number of events in source process:", len(arrival_times_source))

            len_target = len(arrival_times_target)
            h_yy_sec, log_loss_yy = CondH_estimation_yy(
                event_time=[arrival_times_target, arrival_times_source],
                configs=deepcopy(configs),
                seed=seed*(i+1)
            )
            log_yy_losses.append(log_loss_yy)
            
            if  h_yy_sec == float('nan'):
                print(f"Error during TE estimation for run {i+1}. Skipping this run.\n")
                return None, None
        
            print(f'Conditional entropy h_yy for run {i+1}: {h_yy_sec:.5f} nats/sec')
            print(f'Log loss for model yyx: {log_loss_yy:.5f}')

            trial.set_user_attr(f"h_yy_test_sec_run_{i}", h_yy_sec)
            trial.set_user_attr(f"log_loss_yy_run_{i}", log_loss_yy)
            
            H_yy_tests_sec.append(h_yy_sec)
        
        # 3. Return the metric to optimize
        return  np.mean(log_yy_losses)

    return objective

if __name__ == "__main__":
    # Define simulation parameters
    SIMULATION_TIME = 5*60  # seconds
    LAMBDA_X = 30          # events/sec for Poisson process
    seed=42  # For reproducibility

    # The conditional intensity table for process Y
    # The keys are tuples: (is_y_inter_event_time_small, is_x_inter_event_time_small)
    intensity_table = {
        (False, False): 0.03 / 1e-3,  # Y_t > 10ms, X_t > 10ms,
        (False, True): 0.3 / 1e-3 ,   # Y_t > 10ms, X_t <= 10ms
        (True, False): 0.06 / 1e-3,    # Y_t <= 10ms, X_t > 10ms
        (True, True): 0.1 / 1e-3      # Y_t <= 10ms, X_t <= 10ms
    }

    torch.manual_seed(seed)
    np.random.seed(seed)
    time_series_length = SIMULATION_TIME   # in seconds, Length of the time series
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    source_events_list = []
    target_events_list = []
    for i in range(3):
        # Run the simulation
        source_events, target_events, count_table, interval_table = simulate_processes(SIMULATION_TIME, LAMBDA_X, intensity_table, seed+i)
        source_events_list.append(torch.tensor(source_events, dtype=torch.float32))
        target_events_list.append(torch.tensor(target_events, dtype=torch.float32))

        with open(f"./results/log_{seed+i}.txt", 'w') as f:
            with redirect_stdout(f):
                compute_reference(count_table, intensity_table, interval_table, SIMULATION_TIME)

    objective_t = create_objective(target_events_list,source_events_list, 
                        time_series_length, device, seed)

    # Assuming 'objective' function is defined as above
    # ,load_if_exists=True to continue from an existing study
    study = optuna.create_study(directions=["minimize"], storage="sqlite:///db.sqlite3"
                                ,load_if_exists=True, study_name=f"poisson_Hyy_{seed:02d}") # Set direction to 'maximize' for TE,  

    study.optimize(objective_t, n_trials=None) # Run for unlimited trials

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
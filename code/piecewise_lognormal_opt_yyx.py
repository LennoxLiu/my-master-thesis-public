from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from piecewise_lognormal import simulate_processes, compute_reference
import optuna
from entropy_tpp import CondH_estimation_yyx

def create_objective(arrival_times_target, arrival_times_source,
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
                "context_extractor": "gru", #trial.suggest_categorical("context_extractor_yyx", ["gru", "lstm", "mlp"]), # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": trial.suggest_categorical("activation_func_yyx", ["Tanh", "ReLU", "GELU"]),
            },
            "train_config_yyx": {
                "L2_weight": trial.suggest_float("L2_weight_yyx", 1e-5, 2e-3, log=True),          # L2 regularization parameter
                "L_entropy_weight": trial.suggest_float("L_entropy_weight_yyx", 1e-10, 1e-3, log=True),      # Weight for the entropy regularization term
                "L_sep_weight": trial.suggest_float("L_sep_weight_yyx", 1e-10, 1e-3, log=True),               # Weight for the separation regularization term
                "L_scale_weight": trial.suggest_float("L_scale_weight_yyx", 1e-10, 1e-3, log=True),             # Weight for the scale regularization term
                "learning_rate": trial.suggest_float("learning_rate_yyx", 1e-5, 1e-2, log=True),           # Learning rate for Adam optimizer
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
            "history_length": 8,             # in number of bins, Length of the history to use for the model
            "plot_pp": False,  # Whether to plot the PP plots
        
        }
        
        # 2. Run your TE estimation with the suggested hyperparameters
        # You'll need to adapt this part to call the relevant functions in your entropy_tpp.py
        # For instance, if you have a main function that takes these as arguments:
        # te_train, te_val, te_test = calculate_te(param1, param2)
        
        H_yyx_tests_sec = []
        print("Number of events in target process:", len(arrival_times_target))
        print("Number of events in source process:", len(arrival_times_source))
        len_target = len(arrival_times_target)
        log_yyx_losses = []
        for i in range(3):
            h_yyx, log_loss_yyx = CondH_estimation_yyx(
                event_time=[arrival_times_target, arrival_times_source],
                configs=deepcopy(configs),
                seed=seed*(i+1)
            )
            log_yyx_losses.append(log_loss_yyx)
            
            if  h_yyx == float('nan'):
                print(f"Error during TE estimation for run {i+1}. Skipping this run.\n")
                return None, None
            
            h_yyx_sec = h_yyx * len_target / time_series_length
        
            print(f'Conditional entropy h_yyx for run {i+1}: {h_yyx:.5f} nats/event, {h_yyx_sec:.5f} nats/sec')
            print(f'Log loss for model yyx: {log_loss_yyx:.5f}')

            trial.set_user_attr(f"h_yyx_test_sec_run_{i}", h_yyx_sec)
            trial.set_user_attr(f"log_loss_yyx_run_{i}", log_loss_yyx)
            
            H_yyx_tests_sec.append(h_yyx_sec)
        
        # 3. Return the metric to optimize
        return  np.mean(log_yyx_losses)

    return objective

# Define simulation parameters
SIMULATION_TIME = 15*60  # seconds
LAMBDA_X = 30          # events/sec for Poisson process
seed=76  # For reproducibility

# The conditional intensity table for process Y
# The keys are tuples: (is_y_inter_event_time_small, is_x_inter_event_time_small)
# the tuples are mu and sigma of the log-normal distribution
# intensity_table = {
#     (False, False): (-5, 0.5) ,  # Y_t > 10ms, X_t > 10ms, 
#     (False, True): (-7, 2),   # Y_t > 10ms, X_t <= 10ms
#     (True, False): (-3, 0.5),    # Y_t <= 10ms, X_t > 10ms
#     (True, True): (-4, 1.5)      # Y_t <= 10ms, X_t <= 10ms
# }

# morphing factor = 0.5
intensity_table = {
    (False, False): (-6, 1.25) ,  # Y_t > 10ms, X_t > 10ms, 
    (False, True): (-7, 2),   # Y_t > 10ms, X_t <= 10ms
    (True, False): (-3.5, 1),    # Y_t <= 10ms, X_t > 10ms
    (True, True): (-4, 1.5)      # Y_t <= 10ms, X_t <= 10ms
}

# Run the simulation
print(f"Starting simulation for {SIMULATION_TIME} seconds...")
x_events, y_events, count_table, _ = simulate_processes(SIMULATION_TIME, LAMBDA_X, intensity_table, seed)
with open(f"./results/log_{seed}.txt", 'w') as f:
    with redirect_stdout(f):
        compute_reference(count_table, intensity_table, SIMULATION_TIME)

torch.manual_seed(seed)
np.random.seed(seed)
time_series_length = SIMULATION_TIME   # in seconds, Length of the time series
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

objective_t = create_objective(torch.tensor(y_events), torch.tensor(x_events),
                     time_series_length, device, seed)

# Assuming 'objective' function is defined as above
# ,load_if_exists=True to continue from an existing study
study = optuna.create_study(directions=["minimize"], storage="sqlite:///db.sqlite3"
                            ,load_if_exists=True, study_name=f"lognormal_Hyyx_{seed:02d}") # Set direction to 'maximize' for TE,  

study.optimize(objective_t, n_trials=None) # Run for unlimited trials

print("Best trial:")
print(f"  Value: {study.best_value}")
print(f"  Params: {study.best_params}")
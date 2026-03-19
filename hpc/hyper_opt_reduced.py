import os
import numpy as np
import torch
import numpy as np
import torch
from copy import deepcopy
import optuna
from src.te_tpp import Ln_estimation_yy
import argparse
from hpc.hpc_header import get_task_params_reduced, read_event_times_reduced
import json

def create_objective(arrival_times_target,
                     time_series_length, device, seed):
    """
    This outer function creates and returns the actual objective function.
    It takes the data as an argument.
    """
    
    def objective(trial):
        # Suggest hyperparameters

        n_layers_yy = trial.suggest_int("n_layers_yy", 1, 2) # how many hidden layers
        hidden_sizes_yy = []
        for i in range(n_layers_yy):
            # Suggest the size for each hidden layer dynamically
            layer_size = 2** trial.suggest_int(f"hidden_size_yy_l{i}", 2, 8)
            hidden_sizes_yy.append(layer_size)

        configs = {
            "model_config_yy": {
                "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
                "context_size": 2** trial.suggest_int("context_size_yy", 1, 8),  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
                "num_mix_components": 2** trial.suggest_int("num_mix_components_yy", 1, 7),  # 32 Number of components for a mixture model
                "hidden_sizes": hidden_sizes_yy,       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "context_extractor": "lstm", #trial.suggest_categorical("context_extractor_yy", ["gru", "lstm"]), # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": trial.suggest_categorical("activation_func_yy", ["Tanh", "ReLU", "GELU"]),
            },
            "train_config_yy": {
                "L2_weight": trial.suggest_float("L2_weight_yy", 1e-10, 1e-3, log=True),          # L2 regularization parameter
                "L_entropy_weight": trial.suggest_float("L_entropy_weight_yy", 1e-10, 1e-3, log=True),      # Weight for the entropy regularization term
                "L_sep_weight": trial.suggest_float("L_sep_weight_yy", 1e-10, 1e-3, log=True),               # Weight for the separation regularization term
                "L_scale_weight": trial.suggest_float("L_scale_weight_yy", 1e-10, 1e-3, log=True),             # Weight for the scale regularization term
                "learning_rate": trial.suggest_float("learning_rate_yy", 5e-4, 1e-2, log=True),           # Learning rate for Adam optimizer
                "max_epochs": 1000,              # For how many epochs to train
                "display_step": 5,               # Display training statistics after every display_step
                "patience": 20,                  # After how many consecutive epochs without improvement of val loss to stop training
            },
            "data_prep_config":{
                "batch_size": batch_size,          # Number of sequences in a batch
                "shuffle": False,                 # Whether to shuffle the time series before splitting into train/val/test
                "total_time": time_series_length,              # in second, Total time of the sequences
                "verbose": False
            },
            "device": device,
            "verbose": False,  # Whether to print the training statistics
            "plot_histograms": False,  # Whether to plot the conditional histograms
            "history_length": history_length,             # in number of bins, Length of the history to use for the model
            "plot_pp": False,  # Whether to plot the PP plots
        }
        
        # Run TE estimation with the suggested hyperparameters
        
        Ln_yy_tests_sec = []
        log_yy_losses = []

        print("Number of events in target process:", len(arrival_times_target))
        
        len_target = len(arrival_times_target)
        ln_yy, log_loss_yy = Ln_estimation_yy(
            event_time=[arrival_times_target],  # Only target events are needed for the reduced model
            configs=deepcopy(configs),
            seed=seed,
            trial=trial
        )
        log_yy_losses.append(log_loss_yy)
        
        if  ln_yy == float('nan'):
            print(f"Error during Ln_yy estimation.\n")
            return None, None
        
        ln_yy_sec = ln_yy * len_target / time_series_length
    
        print(f'Reduced model ln_yy : {ln_yy_sec:.5f} nats/sec, {ln_yy:.5f} nats/event')
        print(f'Reduced model Log loss : {log_loss_yy:.5f}')

        trial.set_user_attr(f"ln_yy_sec", ln_yy_sec)
        trial.set_user_attr(f"log_loss_yy", log_loss_yy)
        
        Ln_yy_tests_sec.append(ln_yy_sec)
        
        # Return the metric to optimize
        return  np.mean(log_yy_losses)

    return objective



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default="data/event_times_data.h5", help="Path to the HDF5 file containing event times")
    parser.add_argument("--task_id", type=int, required=True, help="Slurm Array Task ID")
    parser.add_argument("--history_length", type=int, default=256, help="History length for TE estimation")
    parser.add_argument("--data_time_length", type=int, default=15*60, help="Total time of the sequences in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training the model")
    args = parser.parse_args()
    
    data_file_path = args.data_file_path
    task_id = args.task_id
    history_length = args.history_length
    data_time_length = args.data_time_length # seconds
    seed=args.seed
    num_trials=args.num_trials
    batch_size=args.batch_size
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("results/opt-reduced", exist_ok=True)

    # Read task parameters from task file
    group_id, neuron_id = get_task_params_reduced('hpc/tasks_reduced.csv', task_id)
    if group_id is not None:
        print(f"Optimize task {task_id}: group_id: {group_id}, neuron_id: {neuron_id}")
        # Check if results/opt already has the result file for this task_id
        result_file = f"opt_reduced_{task_id}_best_config.txt"
        if os.path.exists(result_file):
            print(f"Result file already exists: {result_file}")
            exit(0)  # Exit with code 0 to indicate successful completion, so that Slurm won't reschedule this task
        
        if not os.path.exists(f"results/opt/opt_reduced_{task_id}.db"):
            print(f"Starting optimization for task {task_id}...")
        else:
            print(f"Resuming optimization for task {task_id}...")
    else:
        print(f"Warning: Task ID {task_id} not found in task file. Exiting.")
        exit(0)  # Exit with code 0 to indicate successful completion, so that Slurm won't reschedule this task
    
    # Load event times for the specified group_id and neuron_id
    target_events = read_event_times_reduced('data/event_times_data.h5', group_id, neuron_id)
    target_events = torch.tensor(target_events, dtype=torch.float)

    # Print summary statistics
    print("\n--- Data Summary ---")
    print(f"Total events for target process {len(target_events)}")
    print(f"Data Time: {data_time_length} seconds")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # min_resource=10: Don't prune before epoch 10
    # reduction_factor=3: Standard Hyperband setting
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=500)

    objective_t = create_objective(target_events,
                        data_time_length, device, seed)

    # Assuming 'objective' function is defined as above
    # ,load_if_exists=True to continue from an existing study
    study = optuna.create_study(directions=["minimize"], storage=f"sqlite:///results/opt/opt_reduced_{task_id}.db"
                                ,load_if_exists=True, study_name=f"opt_reduced-model_seed={seed:02d}_task={task_id}",
                                pruner=pruner) 

    study.optimize(objective_t, n_trials=num_trials) # Run for unlimited trials

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    # After optimization, save the best configuration to a JSON/TXT file for easy reference
    # Reconstruct the full config using the best parameters
    best = study.best_params
    n_layers_yy = best["n_layers_yy"]
    hidden_sizes_yy = [2 ** best[f"hidden_size_yy_l{i}"] for i in range(n_layers_yy)]

    best_configs = {
        "model_config_yy": {
            "model_name": "LogNormMix",
            "context_size": 2 ** best["context_size_yy"],
            "num_mix_components": 2 ** best["num_mix_components_yy"],
            "hidden_sizes": hidden_sizes_yy,
            "context_extractor": "lstm", #best["context_extractor_yy"],
            "activation_func": best["activation_func_yy"],
        },
        "train_config_yy": {
            "L2_weight": best["L2_weight_yy"],
            "L_entropy_weight": best["L_entropy_weight_yy"],
            "L_sep_weight": best["L_sep_weight_yy"],
            "L_scale_weight": best["L_scale_weight_yy"],
            "learning_rate": best["learning_rate_yy"],
            "max_epochs": 500,
            "display_step": 5,
            "patience": 20,
        },
        "data_prep_config": {
            "batch_size": batch_size,
            "shuffle": False,
            "total_time": data_time_length,
            "verbose": False
        },
        "device": device,
        "verbose": False,
        "plot_histograms": False,
        "history_length": history_length,  # Use the variable from argparse rather than hardcoding 256
        "plot_pp": False,
    }

    # Save to a readable JSON/TXT file
    config_output_file = f"results/opt-reduced/opt_reduced_{task_id}_best_config.txt"
    with open(config_output_file, "w") as f:
        json.dump(best_configs, f, indent=4)
    print(f"Full best configuration saved to {config_output_file}")

    print(f"Optimization finished.")
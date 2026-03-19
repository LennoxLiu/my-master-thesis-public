import os
import numpy as np
import torch
from copy import deepcopy
import optuna
import argparse
import json

from src.te_tpp import Ln_estimation_yyx 
from hpc.hpc_header import get_task_params_full, read_event_times_full

def create_objective(arrival_times_source, arrival_times_target, device, args):
    """
    Creates and returns the objective function for the full model (xy).
    """
    def objective(trial):
        n_layers_yyx = trial.suggest_int("n_layers_yyx", 1, 2)
        hidden_sizes_yyx = []
        for i in range(n_layers_yyx):
            layer_size = 2 ** trial.suggest_int(f"hidden_size_yyx_l{i}", 2, 8)
            hidden_sizes_yyx.append(layer_size)

        configs = {
            "model_config_yyx": {
                "model_name": "LogNormMix",
                "context_size": 2 ** trial.suggest_int("context_size_yyx", 1, 8),
                "num_mix_components": 2 ** trial.suggest_int("num_mix_components_yyx", 1, 7),
                "hidden_sizes": hidden_sizes_yyx,
                "context_extractor": "lstm", #trial.suggest_categorical("context_extractor_yyx", ["gru", "lstm"]),
                "activation_func": trial.suggest_categorical("activation_func_yyx", ["Tanh", "ReLU", "GELU"]),
            },
            "train_config_yyx": {
                "L2_weight": trial.suggest_float("L2_weight_yyx", 1e-10, 1e-3, log=True),
                "L_entropy_weight": trial.suggest_float("L_entropy_weight_yyx", 1e-10, 1e-3, log=True),
                "L_sep_weight": trial.suggest_float("L_sep_weight_yyx", 1e-10, 1e-3, log=True),
                "L_scale_weight": trial.suggest_float("L_scale_weight_yyx", 1e-10, 1e-3, log=True),
                "learning_rate": trial.suggest_float("learning_rate_yyx", 5e-4, 1e-2, log=True),
                "max_epochs": 1000,
                "display_step": 5,
                "patience": 20,
            },
            "data_prep_config":{
                "batch_size": args.batch_size,
                "shuffle": False,
                "total_time": args.data_time_length,
                "verbose": False
            },
            "device": device,
            "verbose": False,
            "plot_histograms": False,
            "history_length": args.history_length,
            "plot_pp": False,
        }
        
        ln_yyx, log_loss_yyx = Ln_estimation_yyx(
            event_time=[arrival_times_target, arrival_times_source],
            configs=deepcopy(configs),
            seed=args.seed,
            trial=trial
        )
        
        if ln_yyx == float('nan'):
            print(f"Error during Ln_yyx estimation.\n")
            return None
            
        ln_yyx_sec = ln_yyx * len(arrival_times_target) / args.data_time_length
    
        print(f'Full model ln_yyx : {ln_yyx_sec:.5f} nats/sec, {ln_yyx:.5f} nats/event')
        print(f'Full model Log loss : {log_loss_yyx:.5f}')

        trial.set_user_attr("ln_yyx_sec", ln_yyx_sec)
        trial.set_user_attr("log_loss_yyx", log_loss_yyx)
        
        return log_loss_yyx

    return objective

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", type=str, default="data/event_times_data.h5")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--history_length", type=int, default=256)
    parser.add_argument("--data_time_length", type=int, default=15*60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("results/opt-full", exist_ok=True)

    s_group, s_neuron, t_group, t_neuron = get_task_params_full('hpc/tasks_full.csv', args.task_id)
    
    if s_group is None:
        print(f"Warning: Task ID {args.task_id} not found. Exiting.")
        exit(0)

    result_file = f"results/opt-full/opt_full_{args.task_id}_best_config.txt"
    if os.path.exists(result_file):
        print(f"Result file already exists: {result_file}")
        exit(0)
            
    db_path = f"results/opt-full/opt_full_{args.task_id}.db"
    
    source_events, target_events = read_event_times_full(args.data_file_path, s_group, s_neuron, t_group, t_neuron)
    
    if source_events is None or target_events is None:
        print("Missing event data. Exiting.")
        exit(1)

    source_events = torch.tensor(source_events, dtype=torch.float)
    target_events = torch.tensor(target_events, dtype=torch.float)

    print(f"\nSource events: {len(source_events)}")
    print(f"Target events: {len(target_events)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=500)

    objective_t = create_objective(source_events, target_events, device, args)

    study = optuna.create_study(
        directions=["minimize"], 
        storage=f"sqlite:///{db_path}",
        load_if_exists=True, 
        study_name=f"opt_full-model_seed={args.seed}_task={args.task_id}",
        pruner=pruner
    ) 

    study.optimize(objective_t, n_trials=args.num_trials)

    best = study.best_params
    hidden_sizes_yyx = [2 ** best[f"hidden_size_yyx_l{i}"] for i in range(best["n_layers_yyx"])]

    best_configs = {
        "model_config_yyx": {
            "model_name": "LogNormMix",
            "context_size": 2 ** best["context_size_yyx"],
            "num_mix_components": 2 ** best["num_mix_components_yyx"],
            "hidden_sizes": hidden_sizes_yyx,
            "context_extractor": best["context_extractor_yyx"],
            "activation_func": best["activation_func_yyx"],
        },
        "train_config_yyx": {
            "L2_weight": best["L2_weight_yyx"],
            "L_entropy_weight": best["L_entropy_weight_yyx"],
            "L_sep_weight": best["L_sep_weight_yyx"],
            "L_scale_weight": best["L_scale_weight_yyx"],
            "learning_rate": best["learning_rate_yyx"],
            "max_epochs": 500,
            "display_step": 5,
            "patience": 20,
        },
        "data_prep_config": {
            "batch_size": args.batch_size,
            "shuffle": False,
            "total_time": args.data_time_length,
            "verbose": False
        },
        "device": device,
        "verbose": False,
        "plot_histograms": False,
        "history_length": args.history_length,
        "plot_pp": False,
    }

    with open(result_file, "w") as f:
        json.dump(best_configs, f, indent=4)
        
    print(f"Optimization finished. Config saved to {result_file}")
from ast import If
from functools import partial
from tabnanny import verbose
import dpp
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy
from typing import Tuple
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
from collections import deque
from hawkes import simulate_hawkes, simulate_hawkes_onestep, simulate_mutually_exciting_hawkes
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss
import math
import torch.distributions as D
from typing import Tuple, Dict, List
import multiprocessing
import pandas as pd
from sklearn.model_selection import KFold
import pandas as pd

import json
import copy
import os

QUAD_MIN=1e-32 
QUAD_MAX=1e32

MIN_TIME = 1e-16

def save_dict_indented(config_dict, filename):
    """
    Saves a dictionary to a text file with indentation but no comments.

    It handles non-serializable objects like torch.device by converting them to strings.

    Args:
        config_dict (dict): The dictionary to save.
        filename (str): The name of the file to save to.
    """
    os.makedirs(f"./results/", exist_ok=True)

    # Create a deep copy to avoid modifying the original dictionary in memory
    config_to_save = copy.deepcopy(config_dict)

    # --- Pre-process for JSON compatibility ---
    # The json module cannot serialize a torch.device object, so we convert it to a string.
    if 'device' in config_to_save and not isinstance(config_to_save['device'], str):
        config_to_save['device'] = str(config_to_save['device'])

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Use json.dump with the 'indent' parameter for pretty-printing
            json.dump(config_to_save, f, indent=4)
        print(f"Indented configuration successfully saved to {filename}")
    except TypeError as e:
        print(f"Error: A value in the dictionary is not JSON serializable: {e}")
    except IOError as e:
        print(f"Error saving file: {e}")


def gen_poission_event_times(lambda_, T) -> torch.Tensor:
    """
    Generate Poisson event times based on a given rate function.
    
    Args:
        lambdas: Function λ(t) that computes the rate at time t.
        T: Total time (seconds).
        
    Returns:
        A list of event times.
    """
    # Generate homogeneous Poisson events (rate = λ*)
    events = torch.cumsum(torch.distributions.Exponential(lambda_).sample((int(T*lambda_*2),)), dim=0)
    events = events[events < T]

    return events

def plot_histogram(model, dl_test, visualize_samples = 20):
    # Plot the prediction results of the next inter-event time
    with torch.no_grad():
        next_inter_event_times = []
        predicted_next_inter_event_times = []

        for history, target in dl_test: # dl_test is a DataLoader for the test set
            history = history.to(device)
            next_inter_event_times.extend(target.cpu().numpy())
            target = target.to(device)
            predicted_next_inter_event_times.extend(model.sample_next_inter_time_dist(history, num_samples=visualize_samples).cpu().numpy())

    predicted_next_inter_event_times= np.array(predicted_next_inter_event_times)

        # Exclude inf values in the predicted times
    mask = ~np.isinf(predicted_next_inter_event_times).any(axis=1)
    predicted_next_inter_event_times = predicted_next_inter_event_times[mask]
    next_inter_event_times = np.array(next_inter_event_times)[mask]
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 1. Histogram of log sampled next inter-event times for selected indices
    indices = np.random.choice(range(len(predicted_next_inter_event_times)), size=5, replace=False).astype(int)
    print("Visualizing indices: ", indices)
    axs[0].hist(np.log(np.clip(predicted_next_inter_event_times[indices].T, a_min=1e-10, a_max=1e+10)), bins=30)
    axs[0].set_xlabel("Log Sampled Next Inter-Event Time")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram of Log Sampled Next Inter-Event Times (Selected)")

    clipped_predicted_times_log=np.log10(np.clip(predicted_next_inter_event_times[:,0], a_min=1e-10, a_max=None)) # plot the first sample
    actual_times_log= np.log10(next_inter_event_times)
    combined_data = np.concatenate((clipped_predicted_times_log, actual_times_log))
    _, bin_edges = np.histogram(combined_data, bins=100)
    # Overlay histogram of log sampled next inter-event times (all) and log actual next inter-event times

    axs[1].hist(
        actual_times_log, bins=bin_edges, 
        color='darkorange', edgecolor='black', alpha=0.5, label=('Actual(#Events: {})'.format(len(actual_times_log)) )
    )
    axs[1].hist(
        clipped_predicted_times_log, bins=bin_edges,
        color='skyblue', edgecolor='black', alpha=0.5, label=('Sampled(#Events: {})'.format(len(clipped_predicted_times_log)) )
    )
    axs[1].set_xlabel("Log10 Next Inter-Event Time")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Histogram of Log10 Next Inter-Event Times (Sampled vs Actual)")
    axs[1].legend()

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.show()


def get_probabilities(model, dl_test, device) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        probs_data = []
        # If the dataset is too large, we can take only several batches
        for history_batch, target_batch in dl_test:
            history_batch = history_batch.to(device)
            target_batch = target_batch.to(device)

            contexts = model.get_context(history_batch)
            dists = model.get_inter_time_dist(contexts)
            probs = dists.cdf(target_batch)
            probs_data.extend(probs.cpu().numpy())

    
    probs_data = np.array(probs_data)
    probs_data = probs_data[~np.isinf(probs_data) & ~np.isnan(probs_data)]

    # Use the Probability Integral Transform (PIT)
    # The PIT states that if a random variable Y is drawn from a continuous distribution 
    # with a cumulative distribution function (CDF) F, then the new random variable U=F(Y)
    # is uniformly distributed between 0 and 1.

    n = len(probs_data)
    probs_model = torch.arange(0.5, n + 0.5) / n # Uniform quantiles from (0.5/n) to (1-0.5/n)
    probs_data = np.sort(probs_data)

    return torch.tensor(probs_data), probs_model


def plot_pp(
    probability_data_sorted, theoretical_probabilities,
    title: str = 'P-P Plot: Observed Data vs. Model Distribution',
    file_path = "./results/pp_plot/pp_plot.png",
    ax=None
):
    """
    Generates a P-P plot comparing observed data against a fitted
    LogNormalMixtureDistribution.

    """ 
    print("Plotting P-P plot...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    base_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(base_name)[0]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.figure # Get the figure from the provided axes

    n = probability_data_sorted.numel()

    if n == 0:
        print("Warning: No valid observed data points after filtering. Cannot create Q-Q plot.")
        ax.set_title(title + "\n(No valid data to plot)")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Observed Data Quantiles")
        return fig, ax


    if theoretical_probabilities.numel() == 0:
        print("Warning: No valid theoretical probabilities after filtering. Cannot create P-P plot.")
        ax.set_title(title + "\n(No valid theoretical probabilities to plot)")
        ax.set_ylabel("Observed Data Probabilities")
        ax.set_xlabel("Theoretical Probabilities")
        return fig, ax

    # 4. Plotting log P-P
    # Save plotting data to disk before plotting
    plot_data = {
        "theoretical_probabilities": theoretical_probabilities.cpu().numpy().tolist(),
        "probability_data_sorted": probability_data_sorted.cpu().numpy().tolist()
    }
    os.makedirs("./results/pp_plot/", exist_ok=True)
    with open(f"./results/pp_plot/{file_name_no_ext}_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)

    ax.scatter(
        theoretical_probabilities.cpu().numpy(), # Move to CPU for plotting
        probability_data_sorted.cpu().numpy(), # Move to CPU for plotting
        s=10,
        alpha=0.7,
        label="Data Probabilities"
    )

    # Add a 45-degree reference line (y=x)
    # Determine the min/max for the reference line based on the data range
    combined_min = min(theoretical_probabilities.min().item(), probability_data_sorted.min().item())
    combined_max = max(theoretical_probabilities.max().item(), probability_data_sorted.max().item())
    ax.plot(
        [combined_min, combined_max],
        [combined_min, combined_max],
        color='red',
        linestyle='--',
        label='Reference Line (y=x)'
    )

    ax.set_title(title)
    ax.set_ylabel('Observed Data Probabilities')
    ax.set_xlabel('Theoretical Probabilities')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    
    # Add a 45-degree reference line (y=x)
    # Determine the min/max for the reference line based on the data range
    ax.plot(
        [combined_min, combined_max],
        [combined_min, combined_max],
        color='red',
        linestyle='--',
        label='Reference Line (y=x)'
    )

    plt.tight_layout()
    # plt.show()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    print("P-P plot saved to", file_path)

# Event times should NOT insert zero at the beginning
def prepare_dataloaders(
    event_time, configs: dict, seed = None, device = 'cpu'
) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader],int]:
    if configs["verbose"]:
        print('Preparing data...')
    assert isinstance(event_time, list)
    assert all(isinstance(evt, torch.Tensor) for evt in event_time), "All event_time elements must be torch.Tensor"
    assert len(event_time) == 2, "Currently only support two processes for TE estimation"

    target_times = event_time[0][event_time[0]< configs["total_time"]]
    source_times = event_time[1][event_time[1]< configs["total_time"]]

    len_target = len(target_times)

    target_inter_times = torch.diff(target_times) # inter-event times at index i corresponds to event at target_times[i+1]
    source_inter_times = torch.diff(source_times)

    history_len = configs["history_length"]
    
    # We will build three separate lists
    histories_target_list = []
    histories_source_list = []
    time_deltas_list = []
    targets_list = []

    # Initialize the source pointer
    source_idx = 0

    # Iterate through each possible target event
    for target_idx in range(history_len + 1, len(target_times)):
        current_target_time = target_times[target_idx]

        # Advance the source pointer to the correct position
        # We want source_times[source_idx] to be the most recent event strictly before current_target_time
        while source_idx < len(source_times) - 2 and source_times[source_idx + 1] < current_target_time:
            source_idx += 1

        # When source_idx == len(source_times) - 2, (source_idx + 1 = len(source_times) - 1 ) is the last source event,
        # since it must be the most recent event before current_target_time,
        # we do not need to advance further or drop it.

        # 1. NEW CHECK: We now need `history_len` inter-event times from the source.
        # This requires at least `history_len+1` events, so the final index `source_idx`
        # must be at least `history_len`.
        if source_idx >= history_len:
            
            # --- This is a valid sample, construct all parts ---

            # Target value (the inter-event time we want to predict)
            target_val = target_inter_times[target_idx - 1]
            targets_list.append(target_val)

            # Target History: The last `history_len` inter-event times
            history_target = target_inter_times[target_idx - history_len - 1: target_idx - 1]
            histories_target_list.append(history_target)
            
            # Source History: The last `history_len` inter-event times
            history_source = source_inter_times[source_idx - history_len + 1 : source_idx]
            histories_source_list.append(history_source)
            
            # Additional Feature: Time elapsed from the closest source event
            time_since_last_source = current_target_time - source_times[source_idx]

            assert time_since_last_source >= 0.0, "Time since last source event should be non-negative, got {}".format(time_since_last_source)

            time_deltas_list.append(torch.tensor([time_since_last_source]))

    if not targets_list:
        raise ValueError("Could not find any valid sequences with the given history length and data.")

   
    # Stack the lists into final tensors
    targets = torch.stack(targets_list)

    num_small_targets = int((targets < MIN_TIME).sum().item())
    if num_small_targets > 0:
        print(f"Number of targets below MIN_TIME={MIN_TIME}: {num_small_targets}")

    targets = targets.clamp(min=MIN_TIME)  # to avoid zero inter-event times
    histories_target = torch.stack(histories_target_list)
    histories_source = torch.stack(histories_source_list)
    time_deltas = torch.stack(time_deltas_list)

    targets_yy = targets

    histories_source = torch.hstack([histories_source, time_deltas])
    # You can now combine these as needed for your model, for example:
    # Stack the two histories to create a multi-process history tensor
    histories = torch.stack([histories_target, histories_source], dim=2)
    # Shape will be (seq_num, history_length, 2)
    
    # Log-transform the inter-event times for better numerical stability
    histories = torch.log(histories.clamp(min=MIN_TIME))

    assert targets.shape[0] == histories.shape[0], "Targets and history must have the same length"
    seq_num = targets.shape[0]
    # Train/Val/Test split
    indices = np.arange(seq_num)
    if seed is not None:
        np.random.seed(seed)
    if configs["shuffle"]:
        np.random.shuffle(indices)
    else:
        shift = int(seq_num * np.random.rand())
        indices = np.roll(indices, shift)
    train_ratio=0.6
    val_ratio=0.2
    train_end = int(train_ratio * seq_num)  # 60% for training
    val_end = int((train_ratio+val_ratio) * seq_num)    # 20% for validation
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    train_dataset = TensorDataset(histories[train_indices], targets[train_indices])
    val_dataset = TensorDataset(histories[val_indices], targets[val_indices])
    test_dataset = TensorDataset(histories[test_indices], targets[test_indices])
    # get data loader
    dl_train = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
    dl_val = DataLoader(val_dataset, batch_size=configs["batch_size"], shuffle=False)
    dl_test = DataLoader(test_dataset, batch_size=configs["batch_size"], shuffle=False)
    
    # if len(event_time) == 1:
    #     return (dl_train, dl_val, dl_test)
    # else:
    # If we have multiple neurons, return the dataloaders for the first neuron seperately
    dl_train_yy = DataLoader(TensorDataset(histories[train_indices,:,0].unsqueeze(-1), targets_yy[train_indices]), batch_size=configs["batch_size"], shuffle=True)
    dl_val_yy = DataLoader(TensorDataset(histories[val_indices,:,0].unsqueeze(-1), targets_yy[val_indices]), batch_size=configs["batch_size"], shuffle=False)
    dl_test_yy = DataLoader(TensorDataset(histories[test_indices,:,0].unsqueeze(-1), targets_yy[test_indices]), batch_size=configs["batch_size"], shuffle=False)
    return (dl_train, dl_val, dl_test), (dl_train_yy, dl_val_yy, dl_test_yy), len_target

# event time in seconds
def train_tpp_model(dl_train, dl_val, dl_test, configs: dict, seed = None, device = 'cpu', verbose = False) -> Tuple[dpp.models.LogNormMix, float]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Define the model
    if verbose:
        print('Building model...')
    # mean_log_inter_time, std_log_inter_time of target
    log_inter_times = torch.log(dl_train.dataset.tensors[1].clamp(min=MIN_TIME))
    histories = dl_train.dataset.tensors[0]
    if configs["model_config"]["num_processes"] > 1:
        log_inter_times_source = histories[:,:,1].flatten()
        mean_log_inter_time_source = log_inter_times_source.mean().item()
        std_log_inter_time_source = log_inter_times_source.std().item()
    else:
        log_inter_times_source = histories.flatten()
        mean_log_inter_time_source = log_inter_times_source.mean().item()
        std_log_inter_time_source = log_inter_times_source.std().item()
    
    model_class = dpp.models.LogNormMix
    model_name = configs["model_config"]["model_name"]
    model_config = deepcopy(configs["model_config"])
    try:
        del model_config["model_name"] # Remove model_name from config
    except KeyError:
        pass

    model = model_class(
        num_marks=1,  # Number of marks in the TPP, here we use only one mark
        mean_log_inter_time=log_inter_times.mean().item(),
        std_log_inter_time=log_inter_times.std().item(),
        mean_log_inter_time_source=mean_log_inter_time_source,
        std_log_inter_time_source=std_log_inter_time_source,
        history_length=configs["history_length"],
        **model_config
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), weight_decay=configs["train_config"]["L2_weight"], lr=configs["train_config"]["learning_rate"])
    # Use a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.75, patience=5, min_lr=1e-6
    )

    def aggregate_next_loss_over_dataloader(dl):
        total_loss = 0.0
        total_count = 0 # number of sequences in the batch
        with torch.no_grad():
            for history,target in dl:
                history = history.to(device)
                target = target.to(device)
                total_loss += -model.log_prob_next(history,target).sum().item()
                total_count += len(target)
        return total_loss / total_count
    
    
    # Traning
    if verbose:
        print(f'Training model {configs["model_config"]["model_name"]} with {configs["model_config"]["num_processes"]} processes, ')


    impatient = 0
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    training_val_losses = []

    # Combine them into the final loss function.
    L_entropy_weight = configs["train_config"].get("L_entropy_weight", 0.0) # Default to 0 if not set
    L_sep_weight = configs["train_config"].get("L_sep_weight", 0.0) # Default to 0 if not set
    L_scale_weight = configs["train_config"].get("L_scale_weight", 0.0) # Default to 0 if not set
    L_mean_match_weight = configs["train_config"].get("L_mean_match_weight", 0.0) # Default to 0 if not set
   
    for epoch in range(configs["train_config"]["max_epochs"]):
        model.train()
        loss = torch.tensor(0.0)  # Initialize loss to a default value
        total_loss = 0.0
        for history,target in dl_train:
            history = history.to(device)
            target = target.to(device)
            opt.zero_grad()
            # Get the full distribution object, not just the log probability.
            context = model.get_context(history)
            inter_time_dist = model.get_inter_time_dist(context)
            
            # Calculate the Negative Log-Likelihood (NLL) loss as before.
            nll_loss = -inter_time_dist.log_prob(target).mean()
            
            L_weight = torch.tensor(0.0)
            L_scale = torch.tensor(0.0)
            L_sep = torch.tensor(0.0)

            # Calculate the entropy of the mixture weights.
            # The mixture distribution is a Categorical distribution over the components.
            # Its entropy is what we want to maximize.
            mixture_dist = inter_time_dist.base_dist._mixture_distribution
            if L_entropy_weight > 0.0:
                # Encourage higher entropy in the mixture weights
                L_weight = -mixture_dist.entropy().mean() # Average entropy across the batch
            
            if L_sep_weight > 0.0:
                # Encourage larger std of the components to avoid collapse
                L_sep = -torch.std(inter_time_dist.locs, dim=-1).mean()
            
            if L_scale_weight > 0.0:
                # Encourage smaller scale of the components to avoid long tails
                L_scale = torch.mean(inter_time_dist.log_scales.exp())
            
            loss = nll_loss + L_entropy_weight * L_weight + L_sep_weight * L_sep +\
            L_scale_weight * L_scale

            loss.backward()

            total_loss += loss.item() * len(target)
            opt.step()

        total_loss /= len(dl_train.dataset) # normalize by the number of sequences

        model.eval()
        with torch.no_grad():
            loss_val = aggregate_next_loss_over_dataloader(dl_val)
            training_val_losses.append(loss_val)
            
            # Step the scheduler
            scheduler.step(loss_val)

        if (best_loss - loss_val) < 1e-4:
            impatient += 1
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
        else:
            best_loss = loss_val
            best_model = deepcopy(model.state_dict())
            impatient = 0

        if impatient >= configs["train_config"]["patience"]:
            if verbose:
                print(f'Validation loss did not improve for {configs["train_config"]["patience"]} epochs, early stopping at epoch {epoch}.')
            break
        
        if verbose:
            if epoch % configs["train_config"]["display_step"] == 0:
                with torch.no_grad():
                    total_loss = aggregate_next_loss_over_dataloader(dl_train)
                print(f"Epoch {epoch:4d}: loss_train = {total_loss:.3f}, loss_val = {loss_val:.3f}, lr = {opt.param_groups[0]['lr']:.6f}")


    # Evaluation
    model.load_state_dict(best_model)
    model.eval()

    if verbose:
        with torch.no_grad():
            final_loss_train = aggregate_next_loss_over_dataloader(dl_train)
            final_loss_val = aggregate_next_loss_over_dataloader(dl_val)
            final_loss_test = aggregate_next_loss_over_dataloader(dl_test)
        
        print(f'Next inter-event time loss (per event):\n' + \
            f' - Train: {final_loss_train:.5f}\n' + \
            f' - Val:   {final_loss_val:.5f}\n' + \
            f' - Test:  {final_loss_test:.5f}')

    # Calculate logarithmic score (negative log-likelihood on test set)
    with torch.no_grad():
        log_loss = aggregate_next_loss_over_dataloader(dl_test)

    if configs["plot_pp"]:
        data_p, model_p = get_probabilities(model, dl_test, device)
        
        # Plot the P-P plot of the observed data against the fitted distribution
        plot_pp(data_p, model_p,file_path=f"./results/pp_plot/pp_plot_{model_name}_{seed}.png")

    # # Calculate probability loss
    # log_loss = torch.mean(
    #     torch.abs(data_p - model_p)
    # )
    # plot_histogram(model, dl_test, visualize_samples = 20)

    # if verbose:
    #     print(f"Log loss: {log_loss.item():.5f}")

    return model, log_loss


def calculate_entropy_ghq(
    dists: D.TransformedDistribution,
    hermite_roots: torch.Tensor,
    hermite_weights: torch.Tensor
) -> torch.Tensor:
    """
    Calculates entropy for a batch of LogNormalMixtureDistribution objects
    by decomposing the integral over its Gaussian Mixture Model components.

    Args:
        dists: An instance of your LogNormalMixtureDistribution.
        hermite_roots (torch.Tensor): Roots of the Hermite polynomial.
        hermite_weights (torch.Tensor): Weights for GHQ.

    Returns:
        The differential entropy for each mixture distribution in the batch.
    """
    # 1. Deconstruct the mixture distribution
    gmm = dists.base_dist
    component_dist = gmm.component_distribution
    mixture_weights = gmm.mixture_distribution.probs

    a = dists.std_log_inter_time
    b = dists.mean_log_inter_time

    mu_i = component_dist.loc
    sigma_i = component_dist.scale

    t_k = hermite_roots.view(1, 1, -1)
    w_k = hermite_weights.view(1, 1, -1)

    # 2. Perform GHQ for each component via broadcasting
    y_points = mu_i.unsqueeze(-1) + sigma_i.unsqueeze(-1) * math.sqrt(2) * t_k
    
    # Add bounds checking to prevent overflow/underflow
    exponent = a * y_points + b
    exponent = torch.clamp(exponent, min=np.log10(QUAD_MIN), max=np.log10(QUAD_MAX))  # Reasonable bounds for numerical stability
    z_points = torch.exp(exponent) # Shape: [B, K, D]
    
    # Ensure all values are strictly positive and within reasonable bounds for log-normal
    z_points = torch.clamp(z_points, min=QUAD_MIN, max=QUAD_MAX)

    # 3. Evaluate the integrand, using the correct shape convention for log_prob
    batch_size, num_components, degree = z_points.shape
    
    # Flatten points: [B, K, D] -> [B, K*D]
    z_points_flat = z_points.reshape(batch_size, -1)
    
    # <-- FIX: Transpose for log_prob: [B, K*D] -> [K*D, B]
    z_points_swapped = z_points_flat.transpose(0, 1)

    # Output will be [K*D, B]
    log_prob_swapped = dists.log_prob(z_points_swapped)

    # <-- FIX: Transpose back: [K*D, B] -> [B, K*D]
    log_prob_flat = log_prob_swapped.transpose(0, 1)

    # Reshape to the structured grid: [B, K*D] -> [B, K, D]
    log_prob_values = log_prob_flat.reshape(batch_size, num_components, degree)
    
    integrand = -log_prob_values

    # 4. Calculate the integral for each component using GHQ
    component_integrals = (1.0 / math.sqrt(math.pi)) * torch.sum(integrand * w_k, dim=-1)

    # 5. Compute the final entropy by taking the weighted sum
    entropy = torch.sum(mixture_weights * component_integrals, dim=-1)

    return entropy


def EstimateCondEntropy_ExpSinh(
    model,
    dl,
    device,
    degree=256
) -> float:
    """
    Estimates continuous transfer entropy using Tanh-Sinh Quadrature.
    This method is deterministic and serves as an alternative to GHQ.
    """
    H = []

    # 1. Get Tanh-Sinh nodes and weights ONCE and move them to the device
    nodes, weights = exp_sinh_nodes_weights(degree, device)

    model.eval()
    with torch.no_grad():
        for history, _ in dl:
            history = history.to(device)

            # 2. Get the conditional distributions from the models
            context = model.get_context(history)
            dists = model.get_inter_time_dist(context)

            # 3. Calculate the batch of entropies using the Exp-Sinh helper function
            h = calculate_entropy_exp_sinh(dists, nodes, weights)

            H.extend(h.cpu().numpy())

    return float(np.mean(H))


def EstimateContinuousTE_GHQ(
    model_yy,
    model_yyx,
    dl_yyx,
    device,
    degree=128
) -> tuple[float, float, float]:
    """
    Estimates continuous transfer entropy using Gauss-Hermite Quadrature.
    This method is deterministic and often more numerically stable than GLQ.
    """
    te_estimates, H_yy, H_yyx = [], [], []

    # 1. Get Hermite roots and weights ONCE and move them to the device
    x_k_np, w_k_np = hermgauss(degree)
    hermite_roots = torch.from_numpy(x_k_np).float().to(device)
    hermite_weights = torch.from_numpy(w_k_np).float().to(device)

    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, _ in dl_yyx:
            history = history.to(device)

            # 2. Get the conditional distributions from the models
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)

            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)

            # 3. Calculate the batch of entropies using the GHQ helper function
            h_yyx = calculate_entropy_ghq(dists_yyx, hermite_roots, hermite_weights)
            h_yy = calculate_entropy_ghq(dists_yy, hermite_roots, hermite_weights)

            # 4. Calculate transfer entropy for the batch and store results
            te_estimate = h_yy - h_yyx
            te_estimates.extend(te_estimate.cpu().numpy())
            H_yy.extend(h_yy.cpu().numpy())
            H_yyx.extend(h_yyx.cpu().numpy())

    return float(np.mean(te_estimates)), float(np.mean(H_yy)), float(np.mean(H_yyx))


def exp_sinh_nodes_weights(degree: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates nodes and weights for Exp-Sinh Quadrature on (0, inf).

    x = exp( (pi/2) * sinh(t) )
    dx/dt = (pi/2) * cosh(t) * exp( (pi/2) * sinh(t) )

    The integral is approximated by a sum using the trapezoidal rule on the
    transformed function, which converges extremely quickly.

    Args:
        degree (int): The number of points to use (half positive, half negative).
        device (torch.device): The torch device to place the tensors on.

    Returns:
        A tuple containing:
        - nodes (torch.Tensor): The evaluation points `x_k` in (-1, 1).
        - weights (torch.Tensor): The corresponding weights `w_k`.
    """
    # A max value of 4.0 for t, the range of sampling point is roughly [2.4e-19, 4.1e18]
    h = 8.0 / (2 * degree)
    k = torch.arange(-degree, degree + 1, device=device, dtype=torch.float64)

    # Discretize the transformed variable `t`
    t_nodes = k * h

    # The transformation for the nodes is x_k = exp( (pi/2) * sinh(t_k) )
    pi_half_sinh_t = 0.5 * torch.pi * torch.sinh(t_nodes)
    nodes = torch.exp(pi_half_sinh_t)

    # The weights are h * dx/dt evaluated at t_k
    pi_half_cosh_t = 0.5 * torch.pi * torch.cosh(t_nodes)
    weights = h * pi_half_cosh_t * torch.exp(pi_half_sinh_t)

    return nodes.float(), weights.float()


def calculate_entropy_exp_sinh(
    dists: D.TransformedDistribution,
    nodes: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Calculates entropy for a batch of distributions using numerical quadrature.

    This function is adapted for a generic case where the base distribution
    lives on the interval (-1, 1), which is the natural domain for the
    tanh-sinh quadrature rule.

    Args:
        dists: An instance of a distribution transformed from a base on (-1, 1).
        nodes (torch.Tensor): Nodes for quadrature `x_k`.
        weights (torch.Tensor): Weights for quadrature.

    Returns:
        The differential entropy for each distribution in the batch.
    """
    # Use the provided nodes directly as they are the points `x_k`
    x_points = nodes.view(1, -1)  # Shape: [1, num_nodes]
    # x_points = torch.clamp(x_points, min=QUAD_MIN, max=QUAD_MAX)
    
    # 4. Prepare points for batch evaluation
    batch_size = dists.batch_shape[0] if dists.batch_shape else 1
    # Swap axes to match expected input shape for log_prob: [num_nodes, batch_size]
    x_points = x_points.expand(batch_size, -1).transpose(0, 1)

    # 5. Evaluate the necessary components of the integrand
    # a) The function to take the expectation of: -log p(z)
    log_prob_x = dists.log_prob(x_points).transpose(0, 1)
    prob_x = torch.exp(log_prob_x)

    # 6. Form the complete integrand for the expectation: -p(x) * log(p(z))
    # Note: The change of variables formula for entropy is H(Z) = -E_x[-log p(z(x))]
    integrand = prob_x * (-log_prob_x)

    # 7. Calculate the integral by taking the weighted sum
    entropy_batch = torch.sum(integrand * weights.view(1, -1), dim=-1)

    return entropy_batch


def EstimateContinuousTE_ExpSinh(
    model_yy,
    model_yyx,
    dl_yyx,
    device,
    degree=128
) -> tuple[float, float, float]:
    """
    Estimates continuous transfer entropy using Tanh-Sinh Quadrature.
    This method is deterministic and serves as an alternative to GHQ.
    """
    te_estimates, H_yy, H_yyx = [], [], []

    # 1. Get Tanh-Sinh nodes and weights ONCE and move them to the device
    nodes, weights = exp_sinh_nodes_weights(degree, device)

    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, target in dl_yyx:
            history = history.to(device)

            # 2. Get the conditional distributions from the models
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)

            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)

            # 3. Calculate the batch of entropies using the Tanh-Sinh helper function
            h_yyx = calculate_entropy_exp_sinh(dists_yyx, nodes, weights)
            h_yy = calculate_entropy_exp_sinh(dists_yy, nodes, weights)

            # 4. Calculate transfer entropy for the batch and store results
            te_estimate = h_yy - h_yyx
            te_estimates.extend(te_estimate.cpu().numpy())
            H_yy.extend(h_yy.cpu().numpy())
            H_yyx.extend(h_yyx.cpu().numpy())

    return float(np.mean(te_estimates)), float(np.mean(H_yy)), float(np.mean(H_yyx))

def tanh_sinh_nodes_weights_0a_batch(
    degree: int, 
    target_a_batch: torch.Tensor, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates nodes and weights for Tanh-Sinh Quadrature on (0, a) for a
    BATCH of 'a' values.

    CORRECTED VERSION: This version ensures mathematical consistency.
    The weight (which depends on sech^2) is calculated from the *same*
    clipped tanh value that is used to calculate the node,
    using the identity sech^2(z) = 1 - tanh^2(z).
    
    This avoids numerical saturation at the endpoints while keeping the
    (node, weight) pairs consistent.
    
    Args:
        degree (int): The number of points to use.
        target_a_batch (torch.Tensor): A batch of upper bounds [batch_size].
        device (torch.device): The torch device.

    Returns:
        A tuple containing:
        - nodes_batch (torch.Tensor): [batch_size, num_nodes]
        - weights_batch (torch.Tensor): [batch_size, num_nodes]
    """
    h = 8.0 / (2 * degree)
    k = torch.arange(-degree, degree + 1, device=device, dtype=torch.float64)

    # t_nodes shape: [num_nodes]
    t_nodes = k * h
    
    # --- Pre-calculate t-dependent components ---
    pi_half_sinh_t = 0.5 * torch.pi * torch.sinh(t_nodes)
    cosh_t = torch.cosh(t_nodes) # Still needed for dZ/dt
    
    # --- 1. Calculate and Clip tanh ---
    tanh_term = torch.tanh(pi_half_sinh_t)
    tanh_eps = torch.finfo(torch.float64).eps
    
    # This is our master value, used for both node and weight
    tanh_term_clipped = torch.clamp(
        tanh_term, 
        min=-1.0 + tanh_eps,
    ).unsqueeze(0) # Shape: [1, num_nodes]

    # --- 2. Calculate Nodes from clipped value ---
    # a_batch_col shape: [batch_size, 1]
    a_batch_col = target_a_batch.double().unsqueeze(-1)
    
    # (B, 1) * (0.5 * ((1, N) + 1.0)) -> (B, N)
    nodes_batch = (a_batch_col / 2.0) * (tanh_term_clipped + 1.0)

    # --- 3. Calculate Weights from clipped value ---
    
    # sech^2(Z) = 1 - tanh^2(Z)
    # Use the clipped value: 1 - tanh_term_clipped^2
    sech_sq_term = 1.0 - tanh_term_clipped**2  # Shape: [1, num_nodes]
    
    # The full derivative is:
    # dx/dt = (a/2) * sech^2(Z) * (dZ/dt)
    # dx/dt = (a/2) * sech^2_term * ( (pi/2) * cosh(t) )
    
    # (B, 1) * (1, N) * ( (1, N) )
    dx_dt_batch = (a_batch_col / 2.0) * sech_sq_term * \
                  ( (torch.pi / 2.0) * cosh_t.unsqueeze(0) )
    
    weights_batch = h * dx_dt_batch

    return nodes_batch.float(), weights_batch.float()


def calculate_integral_batch(
    dists: D.TransformedDistribution,
    nodes_batch: torch.Tensor,  # Shape: [batch_size, num_nodes]
    weights_batch: torch.Tensor # Shape: [batch_size, num_nodes]
) -> torch.Tensor:
    """
    Calculates the integral for a batch of distributions where each
    has its own set of nodes and weights.

    Args:
        dists: Batch of distributions.
        nodes_batch (torch.Tensor): Nodes `x_k` [batch_size, num_nodes].
        weights_batch (torch.Tensor): Weights `w_k` [batch_size, num_nodes].

    Returns:
        The definite integral for each distribution [batch_size].
    """
    # nodes_batch is [batch_size, num_nodes]
    # log_prob expects [num_nodes, batch_size]
    x_points = nodes_batch.transpose(0, 1)

    # log_prob_x shape: [num_nodes, batch_size]
    log_prob_x = dists.log_prob(x_points)

    # integrand shape: [batch_size, num_nodes]
    integrand = torch.exp(log_prob_x.transpose(0, 1))

    # weights_batch is already [batch_size, num_nodes]

    # Sum( integrand * weights_batch ) over dim=-1
    # (B, N) * (B, N) -> sum(B, N) -> (B,)
    integral_batch = torch.sum(integrand * weights_batch, dim=-1)

    return integral_batch


def Estimate_HazardEntropy_TanhSinh(
    model,
    dl,
    device,
    degree=128
) -> float:
    """
    Estimates the average survival function S(a) by first calculating
    F(a) = integral(p(y|history) dy) from 0 to 'a' and then computing 1.0 - F(a).
    
    This uses Tanh-Sinh Quadrature on (0, a) for each sample.
    """
    ln_hazards = []

    model.eval()
    with torch.no_grad():
        for history, target in dl:
            history = history.to(device)
            # 'target' is our 'a' for the integral
            target = target.to(device)
            target = target.clamp(min=MIN_TIME)  # Avoid zero inter-event times
            
            # Squeeze target in case it's [batch_size, 1]
            # We need it to be [batch_size]
            target_a_batch = target.squeeze() 
            
            # --- Handle case of single-item batch ---
            if target_a_batch.dim() == 0:
                target_a_batch = target_a_batch.unsqueeze(0)

            # 1. Get the conditional distributions
            context = model.get_context(history)
            dists = model.get_inter_time_dist(context)

            # 2. Get nodes and weights for (0, a) *for this batch*
            # Both nodes and weights depend on 'a' and are batched
            nodes_batch, weights_batch = tanh_sinh_nodes_weights_0a_batch(
                degree, 
                target_a_batch, 
                device
            )

            # 3. Calculate survival function S(a) = 1.0 - F(a)
            # F(a) = integral from 0 to 'a' of p(y|history) dy
            S_a_batch = 1.0 - calculate_integral_batch(dists, nodes_batch, weights_batch)

            # 5. Calculate PDF of event at time 'target'
            log_p_batch = dists.log_prob(target)

            ln_hazard = torch.log(S_a_batch) - log_p_batch

            ln_hazards.extend(ln_hazard.cpu().numpy())

    mean_ln_hazard = float(np.mean(ln_hazards))
    return mean_ln_hazard


def EstimateTE_HazardEntropy_MC(
    model_yy,
    model_yyx,
    dl_yyx, # Dataloader providing (history, target)
    device,
    num_samples: int = 10000 # Number of samples for S(tau) estimation
) -> tuple[float, float, float]: # Returns TE, Avg(-log lambda_null), Avg(-log lambda_full)
    """
    Estimates the average TE rate as E[-log lambda_null] - E[-log lambda_full]
    using Monte Carlo integration to estimate the survival function S(tau_i)
    needed for each hazard rate lambda(tau_i).
    """
    ln_hazard_yyx_sum = 0.0
    ln_hazard_yy_sum = 0.0

    model_yy.eval()
    model_yyx.eval()
    data_len = 0
    with torch.no_grad():
        for history, target in dl_yyx: # target is the observed ISI for this step
            history = history.to(device)
            target = target.to(device)
            # Ensure target is positive for calculations
            target = target.clamp(min=MIN_TIME)

            # --- 1. Get Conditional Distributions ---
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)

            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)

            # --- 2. Calculate PDF values at target ---
            log_p_batch_yyx = dists_yyx.log_prob(target)
            log_p_batch_yy = dists_yy.log_prob(target)

            # --- 3. Estimate Survival Functions S(target) using MC ---
            # Draw samples from each distribution
            # Shape: [num_samples, batch_size]
            samples_yyx = dists_yyx.sample(sample_shape=(num_samples,))
            samples_yy = dists_yy.sample(sample_shape=(num_samples,))

            # Compare samples to target for each item in the batch
            # target shape needs to be broadcastable: [1, batch_size]
            target_bc = target.unsqueeze(0) # Add sample dimension

            # Count samples > target for each batch element
            # Shape: [batch_size]
            survival_count_yyx = (samples_yyx > target_bc).sum(dim=0)
            survival_count_yy = (samples_yy > target_bc).sum(dim=0)

            # Estimate S(tau_i) = count / num_samples
            S_a_batch_yyx = survival_count_yyx / num_samples
            S_a_batch_yy = survival_count_yy / num_samples

            # Clamp survival probabilities to avoid division by zero or log(0)
            S_a_batch_yyx = torch.clamp(S_a_batch_yyx, min=MIN_TIME, max=1.0-MIN_TIME)
            S_a_batch_yy = torch.clamp(S_a_batch_yy, min=MIN_TIME, max=1.0-MIN_TIME)

            # 4. Calculate hazard entropies for the batch
            # log(p(target)/S(target)) = log(p(target)) - log(S(target))
            log_hazard_yyx = log_p_batch_yyx - torch.log(S_a_batch_yyx) 
            log_hazard_yy = log_p_batch_yy - torch.log(S_a_batch_yy)

            ln_hazard_yyx_sum += torch.sum(log_hazard_yyx).cpu()
            ln_hazard_yy_sum += torch.sum(log_hazard_yy).cpu()

            data_len += history.size(0)

    mean_ln_hazard_yyx = float(ln_hazard_yyx_sum) / data_len
    mean_ln_hazard_yy = float(ln_hazard_yy_sum) / data_len

    # TE = E[-log lambda_null] - E[-log lambda_full]
    mean_te = mean_ln_hazard_yyx - mean_ln_hazard_yy

    return mean_te, -mean_ln_hazard_yy, -mean_ln_hazard_yyx

def Estimate_HazardEntropy(
    model,
    dl, # Dataloader providing (history, target)
    device,
) -> float: #  Avg(log lambda)
    """
    Estimates the average TE rate as E[-log lambda_null] - E[-log lambda_full]
    using Monte Carlo integration to estimate the survival function S(tau_i)
    needed for each hazard rate lambda(tau_i).
    """
    ln_hazard_sum = 0.0

    model.eval()
    data_len = 0
    with torch.no_grad():
        for history, target in dl: # target is the observed ISI for this step
            history = history.to(device)
            target = target.to(device)
            # Ensure target is positive for calculations
            target = target.clamp(min=MIN_TIME)

            # --- 1. Get Conditional Distributions ---
            context = model.get_context(history)
            dists = model.get_inter_time_dist(context)

            # --- 2. Calculate PDF values at target ---
            log_p = dists.log_prob(target)

            # Estimate S(tau_i) = count / num_samples
            log_s = dists.log_survival_function(target)

            # 4. Calculate hazard entropies for the batch
            # log(p(target)/S(target) = log(p(target)) - log(S(target)) 
            ln_hazard = log_p - log_s

            ln_hazard_sum += torch.sum(ln_hazard).cpu()

            data_len += history.size(0)

    mean_ln_hazard = float(ln_hazard_sum) / data_len

    return mean_ln_hazard


def Estimate_TE_HazardEntropy(
    model_yy,
    model_yyx,
    dl_yy, # Dataloader providing (history, target)
    dl_yyx, # Dataloader providing (history, target)
    device,
) -> tuple[float, float, float]: # Returns TE, Avg(-log lambda_null), Avg(-log lambda_full)
    """
    Estimates the average TE rate as E[-log lambda_null] - E[-log lambda_full]
    using Monte Carlo integration to estimate the survival function S(tau_i)
    needed for each hazard rate lambda(tau_i).
    """
    mean_ln_hazard_yy = Estimate_HazardEntropy(
        model_yy,
        dl_yy,
        device
    )
    mean_ln_hazard_yyx = Estimate_HazardEntropy(
        model_yyx,
        dl_yyx,
        device
    )

    # TE = E[-log lambda_null] - E[-log lambda_full]
    mean_te = mean_ln_hazard_yyx - mean_ln_hazard_yy

    return mean_te, -mean_ln_hazard_yy, -mean_ln_hazard_yyx

def EstimateTE_HazardEntropy_TanhSinh(
    model_yy,  # Kept to mimic signature, but unused
    model_yyx,
    dl_yyx,
    device,
    degree=128
) -> tuple[float, float, float]:
    """
    Estimates the average survival function S(a) by first calculating
    F(a) = integral(p(y|history) dy) from 0 to 'a' and then computing 1.0 - F(a).
    
    This uses Tanh-Sinh Quadrature on (0, a) for each sample.
    """
    ln_hazard_yyx_sum = 0.0
    ln_hazard_yy_sum = 0.0
    data_len = 0
    model_yyx.eval()
    with torch.no_grad():
        for history, target in dl_yyx:
            history = history.to(device)
            # 'target' is our 'a' for the integral
            target = target.to(device)
            target = target.clamp(min=MIN_TIME)  # Avoid zero inter-event times
            
            # Squeeze target in case it's [batch_size, 1]
            # We need it to be [batch_size]
            target_a_batch = target.squeeze() 
            
            # --- Handle case of single-item batch ---
            if target_a_batch.dim() == 0:
                target_a_batch = target_a_batch.unsqueeze(0)

            # 1. Get the conditional distributions
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)
            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)

            # 2. Get nodes and weights for (0, a) *for this batch*
            # Both nodes and weights depend on 'a' and are batched
            nodes_batch, weights_batch = tanh_sinh_nodes_weights_0a_batch(
                degree, 
                target_a_batch, 
                device
            )

            # 3. Calculate survival function S(a) = 1.0 - F(a)
            # F(a) = integral from 0 to 'a' of p(y|history) dy
            S_a_batch_yyx = 1.0 - calculate_integral_batch(dists_yyx, nodes_batch, weights_batch)
            S_a_batch_yy = 1.0 - calculate_integral_batch(dists_yy, nodes_batch, weights_batch)

            # Clamp survival probabilities to avoid division by zero or log(0)
            S_a_batch_yyx = torch.clamp(S_a_batch_yyx, min=MIN_TIME, max=1.0-MIN_TIME)
            S_a_batch_yy = torch.clamp(S_a_batch_yy, min=MIN_TIME, max=1.0-MIN_TIME)

            # 5. Calculate PDF of event at time 'target'
            log_p_batch_yyx = dists_yyx.log_prob(target)
            log_p_batch_yy = dists_yy.log_prob(target)

            ln_hazard_yyx = log_p_batch_yyx - torch.log(S_a_batch_yyx)
            ln_hazard_yy = log_p_batch_yy - torch.log(S_a_batch_yy)

            ln_hazard_yyx_sum += ln_hazard_yyx.sum().item()
            ln_hazard_yy_sum += ln_hazard_yy.sum().item()

            data_len += history.size(0)

    mean_ln_hazard_yyx = float(ln_hazard_yyx_sum) / data_len
    mean_ln_hazard_yy = float(ln_hazard_yy_sum) / data_len
    return mean_ln_hazard_yyx - mean_ln_hazard_yy, -mean_ln_hazard_yy, -mean_ln_hazard_yyx

def sinh_sinh_nodes_weights(degree: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates nodes and weights for Sinh-Sinh (Double-Exponential) Quadrature.

    For an integral over (-inf, inf), we use the double-exponential substitution:
    y = sinh( (pi/2) * sinh(t) )
    dy = cosh( (pi/2) * sinh(t) ) * (pi/2) * cosh(t) dt

    This provides extremely fast convergence for analytic functions.

    Args:
        degree (int): The number of points to use (half positive, half negative).
        device (torch.device): The torch device to place the tensors on.

    Returns:
        A tuple containing:
        - nodes (torch.Tensor): The evaluation points `t_k`.
        - weights (torch.Tensor): The corresponding weights `w_k`.
    """
    # Define the range and step size `h`
    # A max value of 3-4 is usually sufficient due to faster decay.
    h = 8.0 / (2 * degree)
    k = torch.arange(-degree, degree + 1, device=device)
    pi_half = torch.pi / 2.0

    # Nodes (t_k)
    nodes = k * h
    cosh_t = torch.cosh(nodes)
    sinh_t = torch.sinh(nodes)

    nodes = torch.sinh(pi_half * sinh_t)
    # Quadrature weights based on the derivative of the transformation
    # w_k = h * dy/dt
    weights = h * pi_half * cosh_t * torch.cosh(pi_half * sinh_t)

    return nodes, weights


def calculate_entropy_sinh_sinh(
    dists: D.TransformedDistribution,
    nodes: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    Calculates entropy for a batch of LogNormalMixtureDistribution objects
    using true Sinh-Sinh (Double-Exponential) quadrature.

    Args:
        dists: An instance of your LogNormalMixtureDistribution.
        nodes (torch.Tensor): Nodes for Sinh-Sinh quadrature (t_k).
        weights (torch.Tensor): Weights for Sinh-Sinh quadrature.

    Returns:
        The differential entropy for each mixture distribution in the batch.
    """
    # 1. Deconstruct the distribution
    base_dist = dists.base_dist
    a = dists.std_log_inter_time
    b = dists.mean_log_inter_time

    # 2. Perform the change of variable: y_k = sinh( (pi/2) * sinh(t_k) )
    # This is the key change from the previous version.
    pi_half = torch.pi / 2.0
    y_points = nodes.view(1, -1) # Shape: [1, D]

    # 3. Transform these points back to the log-normal domain `z = exp(a*y + b)`
    exponent = a * y_points + b
    # Clamping can still be useful for numerical stability
    exponent = torch.clamp(exponent, min=-20, max=20)
    z_points = torch.exp(exponent)
    z_points = torch.clamp(z_points, min=QUAD_MIN, max=QUAD_MAX)

    # 4. Prepare points for batch evaluation
    batch_size = dists.batch_shape[0]
    y_points_swapped = y_points.expand(batch_size, -1).transpose(0, 1)
    z_points_swapped = z_points.expand(batch_size, -1).transpose(0, 1)

    # 5. Evaluate the necessary components of the integrand
    # a) The function to take the expectation of: -log p(z)
    log_prob_z = dists.log_prob(z_points_swapped).transpose(0, 1) # Shape: [B, D]

    # b) The probability density of the base distribution: p(y)
    log_prob_y = base_dist.log_prob(y_points_swapped).transpose(0, 1) # Shape: [B, D]
    prob_y = torch.exp(log_prob_y)

    # 6. Form the complete integrand: -p(y) * log(p(z))
    integrand = prob_y * (-log_prob_z)

    # 7. Calculate the integral by taking the weighted sum
    # The new weights correctly account for the double-exponential transform
    entropy_batch = torch.sum(integrand * weights.view(1, -1), dim=-1)

    return entropy_batch


def EstimateContinuousTE_SinhSinh(
    model_yy,
    model_yyx,
    dl_yyx,
    device,
    degree=128
) -> tuple[float, float, float]:
    """
    Estimates continuous transfer entropy using Sinh-Sinh Quadrature.
    """
    te_estimates, H_yy, H_yyx = [], [], []

    # 1. Get Sinh-Sinh nodes and weights
    nodes, weights = sinh_sinh_nodes_weights(degree, device)

    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, _ in dl_yyx:
            history = history.to(device)

            # 2. Get the conditional distributions from the models
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)

            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)

            # 3. Calculate entropies using the new Sinh-Sinh helper function
            h_yyx = calculate_entropy_sinh_sinh(dists_yyx, nodes, weights)
            h_yy = calculate_entropy_sinh_sinh(dists_yy, nodes, weights)

            # 4. Calculate transfer entropy and store results
            te_estimate = h_yy - h_yyx
            te_estimates.extend(te_estimate.cpu().numpy())
            H_yy.extend(h_yy.cpu().numpy())
            H_yyx.extend(h_yyx.cpu().numpy())

    return float(np.mean(te_estimates)), float(np.mean(H_yy)), float(np.mean(H_yyx))


def lognormal_entropy_pytorch(
    mu: torch.Tensor, 
    sigma: torch.Tensor, 
    a: float, 
    b: float
) -> torch.Tensor:
    """
    Computes the analytical entropy of a transformed log-normal distribution.
    """
    # Calculate the parameters of the new underlying normal distribution
    mu_prime = a * mu + b
    sigma_prime_sq = a * sigma
    
    # Apply the standard log-normal entropy formula with the new parameters
    entropy = 0.5 * torch.log(2 * torch.pi * torch.e * sigma_prime_sq**2) + mu_prime
    return entropy


def plot_conditional_histograms(
    plot_data: Dict[str, Dict[str, List[float]]],
    bins: int = 100,
    seed: int = 42
):
    """
    Draws eight histograms (4x4 grid) comparing marginal vs. conditional models.
    This version prevents warnings by checking for empty data lists.
    """
    fig, axes = plt.subplots(4, 4, figsize=(30, 21), sharex=False, sharey=False)
    fig.suptitle("Comparison of Predicted Inter-Event Times: Marginal vs. Conditional Models", fontsize=20)


    axes[0, 1].set_title(r"Marginal Model $\tau_Y(i+1) | \tau_Y(i)$", fontsize=14)
    axes[0, 2].set_title(r"Conditional Model $\tau_Y(i+1) | \tau_Y(i), \tau_X(j)$", fontsize=14)
    axes[0, 3].set_title(r"Original Data $\tau_Y(i+1) | \tau_Y(i), \tau_X(j)$", fontsize=14)
    axes[0, 0].set_title(r"Original Data $\tau_Y(i+1) | \tau_Y(i)$", fontsize=14)

    x_maxes = { "LL": 0.2, "LS": 0.2, "SL": 0.2, "SS": 0.2}

    L_data_orig = np.array(plot_data["original"]["LL"]+plot_data["original"]["LS"])
    S_data_orig = np.array(plot_data["original"]["SL"]+plot_data["original"]["SS"])
    
    for i, key in enumerate(["LL", "LS", "SL", "SS"]):
        # Share cols 0 & 1 (ticks will appear on col 1)
        axes[i, 0].sharey(axes[i, 1])
        # Share cols 2 & 3 (ticks will appear on col 3)
        axes[i, 2].sharey(axes[i, 3])

        # --- Left Column: Marginal Model (yy) ---
        ax_yy = axes[i, 1]
        data_yy = np.array(plot_data["yy"][key])
        ax_yy.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        if data_yy.size > 0: # This is shorthand for `if len(data_yy) > 0`
            ax_yy.hist(data_yy[data_yy<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='skyblue', label=f'Samples: {len(data_yy)}, Max: {data_yy.max():.4f}')
            # ax_yy.hist(data_yy, bins=bins, alpha=0.75, density=True, color='skyblue', label=f'Samples: {len(data_yy)}, Max: {data_yy.max():.4f}')
        ax_yy.legend() # Always call legend to show "Samples: 0" if empty
        ax_yy.grid(axis='y', linestyle='--', alpha=0.7)
        ax_yy.set_ylabel(key, fontsize=12, labelpad=10)

        # --- Right Column: Conditional Model (yyx) ---
        ax_yyx = axes[i, 2]
        data_yyx = np.array(plot_data["yyx"][key])
        ax_yyx.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        if data_yyx.size > 0:
            ax_yyx.hist(data_yyx[data_yyx<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='salmon', label=f'Samples: {len(data_yyx)}, Max: {data_yyx.max():.4f}')
            # ax_yyx.hist(data_yyx, bins=bins, alpha=0.75, density=True, color='salmon', label=f'Samples: {len(data_yyx)}, Max: {data_yyx.max():.4f}')
        ax_yyx.legend()
        ax_yyx.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Third Column: Original Data ---
        ax_orig = axes[i, 3]
        data_orig = np.array(plot_data["original"][key])
        ax_orig.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        if data_orig.size > 0:
            ax_orig.hist(data_orig[data_orig<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='lightgreen', label=f'Samples: {len(data_orig)}, Max: {data_orig.max():.4f}')
        ax_orig.legend()
        ax_orig.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Fourth Column: Original Data (Marginal) ---
        ax_orig_marginal = axes[i, 0]
        ax_orig_marginal.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        if key in ["LL", "LS"]:
            if L_data_orig.size > 0:
                ax_orig_marginal.hist(L_data_orig[L_data_orig<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='mediumpurple', label=f'Samples: {len(L_data_orig)}, Max: {L_data_orig.max():.4f}')
        if key in ["SL", "SS"]:
            if S_data_orig.size > 0:
                ax_orig_marginal.hist(S_data_orig[S_data_orig<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='plum', label=f'Samples: {len(S_data_orig)}, Max: {S_data_orig.max():.4f}')
        ax_orig_marginal.legend()
        ax_orig_marginal.grid(axis='y', linestyle='--', alpha=0.7)

        # Set the key as the title for each row
        axes[i, 0].set_ylabel(key, fontsize=12, labelpad=10, rotation=0, ha='right', va='center')

    axes[3, 0].set_xlabel("Predicted $\\tau_Y$", fontsize=12)
    axes[3, 1].set_xlabel("Predicted $\\tau_Y$", fontsize=12)
    axes[3, 2].set_xlabel("Original $\\tau_Y$", fontsize=12)
    axes[3, 3].set_xlabel("Original $\\tau_Y$", fontsize=12)

    fig.text(0.07, 0.5, 'Counts', va='center', rotation='vertical', fontsize=14)

    os.makedirs(f"./results/hists/", exist_ok=True)
    plt.savefig(f"./results/hists/cond_hists_{seed}.png", dpi=300)
    # plt.show()
    plt.close()


def collect_plotting_data(
    model_yy,
    model_yyx,
    dl_yyx,
    device,
    log_threshold = np.log(0.01),
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collects conditional prediction samples from both models for plotting histograms.
    This function is separate from the TE estimation.

    Args:
        model_yy: Model for the target process Y's history (marginal).
        model_yyx: Model for Y conditioned on source X (conditional).
        dl_yyx: DataLoader providing history batches.

    Returns:
        A nested dictionary with prediction samples, structured for the plotting function:
          {'yyx': {'LL': [...], 'LS': [...], ...},
           'yy':  {'LL': [...], 'LS': [...], ...}}
    """
    plot_data = {
        "yyx": {"LL": [], "LS": [], "SL": [], "SS": []}, # Conditional model P(Y|Y,X)
        "yy":  {"LL": [], "LS": [], "SL": [], "SS": []},  # Marginal model P(Y|Y)
        "original": {"LL": [], "LS": [], "SL": [], "SS": []}  # Original data for reference
    }
    
    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, target in dl_yyx: # We don't need the 'target' here
            history = history.to(device)
            target = target.to(device)
            
            # Conditional model P(Y|Y,X)
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)
            samples_yyx = dists_yyx.sample(sample_shape=(10,))

            # Marginal model P(Y|Y)
            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)
            samples_yy = dists_yy.sample(sample_shape=(10,))

            # --- Check conditions and store the samples ---
            cond0_large = history[:, -1, 0] > log_threshold
            cond1_large = history[:, -2, 1] > log_threshold
            
            masks = {
                "LL": cond0_large & cond1_large,
                "LS": cond0_large & ~cond1_large,
                "SL": ~cond0_large & cond1_large,
                "SS": ~cond0_large & ~cond1_large
            }


            # Store predictions based on which condition was met
            for key, mask in masks.items():
                if mask.any():
                    plot_data["yyx"][key].extend(samples_yyx[:,mask].flatten().cpu().numpy())
                    plot_data["yy"][key].extend(samples_yy[:,mask].flatten().cpu().numpy())
                    plot_data["original"][key].extend(target[mask].flatten().cpu().numpy())

    # for model_key in plot_data:
    #     for cond_key in plot_data[model_key]:
    #         print(f"  {model_key} - {cond_key}: {len(plot_data[model_key][cond_key])} samples")
    return plot_data


def collect_plotting_data_CoTETE(
    model_yy,
    model_yyx,
    dl_yyx,
    device,
    log_threshold = np.log(1),
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collects conditional prediction samples from both models for plotting histograms.
    This function is separate from the TE estimation.

    Args:
        model_yy: Model for the target process Y's history (marginal).
        model_yyx: Model for Y conditioned on source X (conditional).
        dl_yyx: DataLoader providing history batches.

    Returns:
        A nested dictionary with prediction samples, structured for the plotting function:
          {'yyx': {'Large': [...], 'Small': [...], ...},
           'yy':  {'Large': [...], 'Small': [...], ...}}
    """
    plot_data = {
        "yyx": {"Large": [], "Small": []}, # Conditional model P(Y|Y,X)
        "yy":  {"Large": [], "Small": []},  # Marginal model P(Y|Y)
        "original": {"Large": [], "Small": []}  # Original data for reference
    }
    
    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, target in dl_yyx: # We don't need the 'target' here
            history = history.to(device)
            target = target.to(device)

            # --- Generate one sample from each model for each event ---
            # NOTE: We only need one sample per event for the histogram (sample_shape=(1,))
            
            # Conditional model P(Y|Y,X)
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)
            samples_yyx = dists_yyx.sample(sample_shape=(10,))

            # Marginal model P(Y|Y)
            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)
            samples_yy = dists_yy.sample(sample_shape=(10,))

            # --- Check conditions and store the samples ---
            cond_large = history[:, -1, 1] > log_threshold
            
            masks = {
               "Large": cond_large,
               "Small": ~cond_large
            }

            # Store predictions based on which condition was met
            for key, mask in masks.items():
                if mask.any():
                    plot_data["yyx"][key].extend(samples_yyx[:,mask].flatten().cpu().numpy())
                    plot_data["yy"][key].extend(samples_yy[:,mask].flatten().cpu().numpy())
                    plot_data["original"][key].extend(target[mask].flatten().cpu().numpy())

    # for model_key in plot_data:
    #     for cond_key in plot_data[model_key]:
    #         print(f"  {model_key} - {cond_key}: {len(plot_data[model_key][cond_key])} samples")
    return plot_data


def plot_conditional_histograms_CoTETE(
    plot_data: Dict[str, Dict[str, List[float]]],
    bins: int = 100,
    seed: int = 42
):
    """
    Draws eight histograms (4x2 grid) comparing marginal vs. conditional models.
    This version prevents warnings by checking for empty data lists.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 16), sharex=False, sharey=False)
    fig.suptitle("Comparison of Predicted Inter-Event Times: Marginal vs. Conditional Models", fontsize=20)


    axes[0, 0].set_title(r"Marginal Model $p(\tau_Y(i+1) | \tau_Y(i))$", fontsize=14)
    axes[0, 1].set_title(r"Conditional Model $p(\tau_Y(i+1) | \tau_Y(i), \tau_X(j))$", fontsize=14)
    axes[0, 2].set_title(r"Original Data $p(\tau_Y(i+1))$", fontsize=14)

    x_maxes = { "Large": 5, "Small": 5 }
    for i, key in enumerate(["Large", "Small"]):
        # --- Left Column: Marginal Model (yy) ---
        ax_yy = axes[i, 0]
        data_yy = np.array(plot_data["yy"][key])
        ax_yy.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        
        # --- FIX: Check if data list is not empty before plotting ---
        if data_yy.size > 0: # This is shorthand for `if len(data_yy) > 0`
            ax_yy.hist(data_yy[data_yy<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='skyblue', label=f'Samples: {len(data_yy)}, Max: {data_yy.max():.4f}')
            # ax_yy.hist(data_yy, bins=bins, alpha=0.75, density=True, color='skyblue', label=f'Samples: {len(data_yy)}, Max: {data_yy.max():.4f}')
        
        ax_yy.legend() # Always call legend to show "Samples: 0" if empty
        ax_yy.grid(axis='y', linestyle='--', alpha=0.7)
        ax_yy.set_ylabel(key, fontsize=12, labelpad=10)

        # --- Right Column: Conditional Model (yyx) ---
        ax_yyx = axes[i, 1]
        data_yyx = np.array(plot_data["yyx"][key])
        ax_yyx.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        
        # --- FIX: Check if data list is not empty before plotting ---
        if data_yyx.size > 0:
            ax_yyx.hist(data_yyx[data_yyx<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='salmon', label=f'Samples: {len(data_yyx)}, Max: {data_yyx.max():.4f}')
            # ax_yyx.hist(data_yyx, bins=bins, alpha=0.75, density=True, color='salmon', label=f'Samples: {len(data_yyx)}, Max: {data_yyx.max():.4f}')

        ax_yyx.legend()
        ax_yyx.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Third Column: Original Data ---
        ax_orig = axes[i, 2]
        data_orig = np.array(plot_data["original"][key])
        ax_orig.set_xlim(0, x_maxes[key]) # Set x-axis limit based on pre-calculated max values
        if data_orig.size > 0:
            ax_orig.hist(data_orig[data_orig<x_maxes[key]], bins=bins, alpha=0.75, density=True, color='lightgreen', label=f'Samples: {len(data_orig)}, Max: {data_orig.max():.4f}')
        
        ax_orig.legend()
        ax_orig.grid(axis='y', linestyle='--', alpha=0.7)

    axes[1, 0].set_xlabel("Predicted $\\tau_Y$", fontsize=12)
    axes[1, 1].set_xlabel("Predicted $\\tau_Y$", fontsize=12)
    axes[1, 2].set_xlabel("Original $\\tau_Y$", fontsize=12)

    fig.text(0.07, 0.5, 'Counts', va='center', rotation='vertical', fontsize=14)

    os.makedirs(f"./results/hists/", exist_ok=True)
    plt.savefig(f"./results/hists/cond_hists_{seed}.png", dpi=300)
    # plt.show()
    plt.close()

def EstimateContinuousTE_MC(
    model_yy, 
    model_yyx, 
    dl_yyx, 
    device,
    num_samples: int = 5000
) -> Tuple[float, float, float]:
    """
    Estimate the continuous transfer entropy between two processes using the model.
    (This is the original, unmodified version).
    """
    te_estimates, H_yy, H_yyx = [], [], []
    model_yy.eval()
    model_yyx.eval()
    with torch.no_grad():
        for history, target in dl_yyx:
            history = history.to(device)
            
            # Get distributions and samples for TE calculation
            context_yyx = model_yyx.get_context(history)
            dists_yyx = model_yyx.get_inter_time_dist(context_yyx)
            samples_yyx = dists_yyx.sample(sample_shape=(num_samples,)).clamp(min=QUAD_MIN)
            log_prob_yyx = dists_yyx.log_prob(samples_yyx)

            context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
            dists_yy = model_yy.get_inter_time_dist(context_yy)
            samples_yy = dists_yy.sample(sample_shape=(num_samples,)).clamp(min=QUAD_MIN)
            log_prob_yy = dists_yy.log_prob(samples_yy)

            # Safely calculate entropies
            mask_yy = torch.isfinite(log_prob_yy)
            h_yy = -((torch.where(mask_yy, log_prob_yy, 0.0).sum(dim=0)) / mask_yy.sum(dim=0).clamp(min=1)).cpu().numpy()
            mask_yyx = torch.isfinite(log_prob_yyx)
            h_yyx = -((torch.where(mask_yyx, log_prob_yyx, 0.0).sum(dim=0)) / mask_yyx.sum(dim=0).clamp(min=1)).cpu().numpy()

            te_estimate = h_yy - h_yyx
            
            te_estimates.extend(te_estimate)
            H_yy.extend(h_yy)
            H_yyx.extend(h_yyx)

    return float(np.mean(te_estimates)), float(np.mean(H_yy)), float(np.mean(H_yyx))
    
def GetAnalyticalReferenceTE(
        model_yy, 
        model_yyx, 
        dl_yyx, 
        configs
    ) -> Tuple[float, float, float]:
        """
        Calculates the analytical Transfer Entropy as a reference, assuming the models
        use only a single log-normal component (num_mix_components=1).
        """
        # CRITICAL: This function is only valid if the models output a single distribution.
        assert model_yyx.num_mix_components == 1, "model_yyx must have num_mix_components=1 for analytical solution."
        assert model_yy.num_mix_components == 1, "model_yy must have num_mix_components=1 for analytical solution."

        te_estimates, H_yy, H_yyx = [], [], []

        model_yy.eval()
        model_yyx.eval()
        with torch.no_grad():
            for history, target in dl_yyx:
                history = history.to(configs["device"])

                # --- Get parameters for H(Y'|Y,X) ---
                context_yyx = model_yyx.get_context(history)
                dists_yyx = model_yyx.get_inter_time_dist(context_yyx)
                
                # Extract mu and sigma for the single log-normal
                # .squeeze(-1) removes the unnecessary 'num_components' dimension
                mu_yyx = dists_yyx.locs.squeeze(-1)
                sigma_yyx = torch.exp(dists_yyx.log_scales.squeeze(-1))

                # --- Get parameters for H(Y'|Y) ---
                context_yy = model_yy.get_context(history[:, :, 0].unsqueeze(-1))
                dists_yy = model_yy.get_inter_time_dist(context_yy)
                
                mu_yy = dists_yy.locs.squeeze(-1)
                sigma_yy = torch.exp(dists_yy.log_scales.squeeze(-1))
                # mean_log_inter_time, std_log_inter_time are the parameters for config_yyx and config_yy
                # --- Calculate entropies using the analytical formula ---
                h_yyx = lognormal_entropy_pytorch(mu_yyx, sigma_yyx,a=model_yyx.get_std_log_inter_time(), b=model_yyx.get_mean_log_inter_time())
                h_yy = lognormal_entropy_pytorch(mu_yy, sigma_yy,a=model_yy.get_std_log_inter_time(), b=model_yy.get_mean_log_inter_time())

                te_estimate = h_yy - h_yyx

                te_estimates.extend(te_estimate.cpu().numpy())
                H_yy.extend(h_yy.cpu().numpy())
                H_yyx.extend(h_yyx.cpu().numpy())

        return float(np.mean(te_estimates)), float(np.mean(H_yy)), float(np.mean(H_yyx))

def CondH_estimation_yy(event_time, configs: dict, seed: int = 42):
    """
    Estimate the conditional entropy H(Y'|Y) of a temporal point process (TPP) using neural models."""

    data_prep_config = deepcopy(configs["data_prep_config"])
    data_prep_config["history_length"] = configs["history_length"]
    # Prepare data loaders
    _, dls_yy, len_target  = prepare_dataloaders(
        event_time=event_time, 
        configs=data_prep_config, 
        seed=seed,
        device=configs["device"]
    )

    config_yy = {}
    # (number of neurons, number of events)
    config_yy["model_config"] = deepcopy(configs["model_config_yy"])
    config_yy["model_config"]["num_processes"] = 1 
    config_yy["train_config"] = deepcopy(configs["train_config_yy"]) # Copy train config to each model
    config_yy["history_length"] = configs["history_length"]
    config_yy["plot_pp"] = configs["plot_pp"]

    model_yy, log_loss_yy = train_tpp_model(*dls_yy, configs=config_yy, seed=seed, device=configs["device"], verbose = configs["verbose"])
    _, _, dl_test_yy = dls_yy
    # H_yy_test = EstimateCondEntropy_ExpSinh(model_yy, dl_test_yy, device=configs["device"], degree=256)
    # H_yy_test = Estimate_HazardEntropy_TanhSinh(model_yy, dl_test_yy, device=configs["device"], degree=256)
    H_yy_test = Estimate_HazardEntropy(model_yy, dl_test_yy, device=configs["device"])

    return H_yy_test * len_target / configs["data_prep_config"]["total_time"], log_loss_yy


def CondH_estimation_yyx(event_time, configs: dict, seed: int = 42):
    """
    Estimate the conditional entropy H(Y'|Y,X) of a temporal point process (TPP) using neural models."""
    data_prep_config = deepcopy(configs["data_prep_config"])
    data_prep_config["history_length"] = configs["history_length"]
    # Prepare data loaders
    dls_yyx, _, len_target = prepare_dataloaders(
        event_time=event_time, 
        configs=data_prep_config, 
        seed=seed,
        device=configs["device"]
    )

    config_yyx = {}
    # (number of neurons, number of events)
    config_yyx["model_config"] = deepcopy(configs["model_config_yyx"])
    config_yyx["model_config"]["num_processes"] = len(event_time)  # Number of neurons in the TPP, here we use only one neuron
    config_yyx["train_config"] = deepcopy(configs["train_config_yyx"]) # Copy train config to each model
    config_yyx["history_length"] = configs["history_length"]
    config_yyx["plot_pp"] = configs["plot_pp"]
    
    model_yyx, log_loss_yyx = train_tpp_model(*dls_yyx, configs=config_yyx, seed=seed, device=configs["device"], verbose = configs["verbose"])
    _, _, dl_test_yyx = dls_yyx
    # H_yyx_test = EstimateCondEntropy_ExpSinh(model_yyx, dl_test_yyx, device=configs["device"], degree=256)
    # H_yyx_test = Estimate_HazardEntropy_TanhSinh(model_yyx, dl_test_yyx, device=configs["device"], degree=256)
    H_yyx_test = Estimate_HazardEntropy(model_yyx, dl_test_yyx, device=configs["device"])

    return H_yyx_test * len_target / configs["data_prep_config"]["total_time"], log_loss_yyx

def TE_estimation_tpp(event_time, configs: dict, seed: int = 42):
    """
    Estimate the transfer entropy (TE) between two temporal point processes (TPPs) using neural models.
    This function prepares data loaders, trains two TPP models (one with and one without access to the source process),
    and estimates the transfer entropy by comparing the log-probabilities of inter-event times under both models.
    Args:
        event_time: Input event time data for the processes in seconds. It can be a tuple of (histories, targets) for single process,
                    or a list of event times for multiple processes. When using a list, the first tensor in the list is considered the target process,
                    and the rest are source processes.
        configs (dict): Configuration dictionary containing model and training parameters.
        seed (optional): Random seed for reproducibility.
    Returns:
        tuple:
            - (TE_test, H_yy_test, H_yyx_test): Estimated transfer entropy values per second for the train, validation, and test sets.
            - (log_loss_yy, log_loss_yyx): Quantile losses for the two trained models.
    Raises:
        Returns NaN values and an error message in case of exceptions during model training or TE estimation.
    Notes:
        - The function assumes the existence of `prepare_dataloaders`, `train_tpp_model`, and the required model methods.
        - Transfer entropy is estimated by sampling inter-event times and comparing log-probabilities under both models.
    
    """
    
    data_prep_config = deepcopy(configs["data_prep_config"])
    data_prep_config["history_length"] = configs["history_length"]
    # Read-only to configs
    dls_yyx, dls_yy, len_target  = prepare_dataloaders(
        event_time=event_time,
        configs=data_prep_config,
        seed=seed,
        device=configs["device"]
    )

    config_yy = {}
    # Set model configuration parameters
    # the max_bin_count is passed to model in the config
    config_yy["model_config"] = deepcopy(configs["model_config_yy"])
    config_yy["model_config"]["num_processes"] = 1  # Number of neurons in the TPP, here we use only one neuron
    config_yy["train_config"] = configs["train_config_yy"] # Copy train config to each model
    config_yy["history_length"] = configs["history_length"]
    config_yy["plot_pp"] = configs["plot_pp"]

    config_yyx = {}
    # (number of neurons, number of events)
    config_yyx["model_config"] = deepcopy(configs["model_config_yyx"])
    config_yyx["model_config"]["num_processes"] = len(event_time)  # Number of neurons in the TPP, here we use only one neuron
    config_yyx["train_config"] = configs["train_config_yyx"] # Copy train config to each model
    config_yyx["history_length"] = configs["history_length"]
    config_yyx["plot_pp"] = configs["plot_pp"]

    model_yy, log_loss_yy = train_tpp_model(*dls_yy, configs=config_yy, seed=seed, device=configs["device"], verbose = configs["verbose"])
    model_yyx, log_loss_yyx = train_tpp_model(*dls_yyx, configs=config_yyx, seed=seed, device=configs["device"], verbose = configs["verbose"])

    dl_train_yyx, dl_val_yyx, dl_test_yyx = dls_yyx
    dl_train_yy, dl_val_yy, dl_test_yy = dls_yy
    
    # TE_test_mc, H_yy_test_mc, H_yyx_test_mc = EstimateContinuousTE_MC(model_yy, model_yyx, dl_test_yyx, device=configs["device"], num_samples=configs["num_mc_samples"])
    # print(f'[MC] Transfer Entropy (nats/event):\n'
    #     f' - H_yy_test: {H_yy_test_mc}\n'
    #     f' - H_yyx_test: {H_yyx_test_mc}\n'
    #     f' - TE_test:  {TE_test_mc}')
    
    # # Use GHQ for faster and more accurate estimation
    # TE_test_ghq, H_yy_test_ghq, H_yyx_test_ghq = EstimateContinuousTE_GHQ(model_yy, model_yyx, dl_test_yyx, device=configs["device"], degree=256)
    # print(f'[GHQ] Transfer Entropy (nats/event):\n'
    #     f' - H_yy_test: {H_yy_test_ghq}\n'
    #     f' - H_yyx_test: {H_yyx_test_ghq}\n'
    #     f' - TE_test:  {TE_test_ghq}')

    # # Use Sinh-Sinh as an alternative deterministic method
    # TE_test_sinhsinh, H_yy_test_sinhsinh, H_yyx_test_sinhsinh = EstimateContinuousTE_SinhSinh(model_yy, model_yyx, dl_test_yyx, device=configs["device"], degree=256)
    # print(f'[Sinh-Sinh] Transfer Entropy (nats/event):\n'
    #     f' - H_yy_test: {H_yy_test_sinhsinh}\n'
    #     f' - H_yyx_test: {H_yyx_test_sinhsinh}\n'
    #     f' - TE_test:  {TE_test_sinhsinh}')
    
    # Use Exp-Sinh as an alternative deterministic method
    # TE_test_ExpSinh, H_yy_test_ExpSinh, H_yyx_test_ExpSinh = EstimateContinuousTE_ExpSinh(model_yy, model_yyx, dl_test_yyx, device=configs["device"], degree=2560)
    # print(f'[Exp-Sinh] Transfer Entropy (nats/event):\n'
    #     f' - ln_hazard_yy_test: {H_yy_test_ExpSinh:.5f}\n'
    #     f' - ln_hazard_yyx_test: {H_yyx_test_ExpSinh:.5f}\n'
    #     f' - TE_test:  {TE_test_ExpSinh:.5f}\n')

    # if model_yy.num_mix_components == 1 and model_yyx.num_mix_components == 1:
    #     TE_test, H_yy_test, H_yyx_test = GetAnalyticalReferenceTE(model_yy, model_yyx, dl_test_yyx, configs)
    #     print(f'[Analytical] Transfer Entropy (nats/event):\n'
    #         f' - ln_hazard_yy_test: {H_yy_test:.5f}\n'
    #         f' - ln_hazard_yyx_test: {H_yyx_test:.5f}\n'
    #         f' - TE_test:  {TE_test:.5f}')
    

    # TE_hazard_TanhSinh, H_yy_hazard_TanhSinh, H_yyx_hazard_TanhSinh = EstimateTE_HazardEntropy_TanhSinh(model_yy, model_yyx, dl_test_yyx, device=configs["device"], degree=256)
    # print(f'[Hazard Tanh-Sinh] Transfer Entropy (nats/event):\n'
    #     f' - ln_hazard_yy_test: {H_yy_hazard_TanhSinh:.5f}\n'
    #     f' - ln_hazard_yyx_test: {H_yyx_hazard_TanhSinh:.5f}\n'
    #     f' - TE_test:  {TE_hazard_TanhSinh:.5f}\n')
    
    # TE_hazard_MC, H_yy_hazard_MC, H_yyx_hazard_MC = EstimateTE_HazardEntropy_MC(model_yy, model_yyx, dl_test_yyx, device=configs["device"], num_samples=configs["num_mc_samples"])
    # print(f'[Hazard MC] Transfer Entropy (nats/event):\n'
    #     f' - ln_hazard_yy_test: {H_yy_hazard_MC:.5f}\n'
    #     f' - ln_hazard_yyx_test: {H_yyx_hazard_MC:.5f}\n'
    #     f' - TE_test:  {TE_hazard_MC:.5f}\n')

    TE_hazard, H_yy_hazard, H_yyx_hazard = Estimate_TE_HazardEntropy(model_yy, model_yyx, dl_test_yy, dl_test_yyx, device=configs["device"])
    print(f'[Hazard Basic] Transfer Entropy (nats/event):\n'
        f' - ln_hazard_yy_test: {H_yy_hazard:.5f}\n'
        f' - ln_hazard_yyx_test: {H_yyx_hazard:.5f}\n'
        f' - TE_test:  {TE_hazard:.5f}\n')

    TE_hazard_train, H_yy_hazard_train, H_yyx_hazard_train = Estimate_TE_HazardEntropy(model_yy, model_yyx, dl_train_yy, dl_train_yyx, device=configs["device"])
    print(f'[Hazard Basic] Transfer Entropy (nats/event):\n'
        f' - ln_hazard_yy_train: {H_yy_hazard_train:.5f}\n'
        f' - ln_hazard_yyx_train: {H_yyx_hazard_train:.5f}\n'
        f' - TE_train:  {TE_hazard_train:.5f}\n')

    TE_hazard_val, H_yy_hazard_val, H_yyx_hazard_val = Estimate_TE_HazardEntropy(model_yy, model_yyx, dl_val_yy, dl_val_yyx, device=configs["device"])
    print(f'[Hazard Basic] Transfer Entropy (nats/event):\n'
        f' - ln_hazard_yy_val: {H_yy_hazard_val:.5f}\n'
        f' - ln_hazard_yyx_val: {H_yyx_hazard_val:.5f}\n'
        f' - TE_val:  {TE_hazard_val:.5f}\n')
    
    if configs["plot_histograms"]:
        print("Collecting data for conditional histograms...")
        data_for_plotting = collect_plotting_data(model_yy, model_yyx, dl_test_yyx, configs["device"], log_threshold = np.log(0.01))
        # data_for_plotting = collect_plotting_data_CoTETE(model_yy, model_yyx, dl_test_yyx, configs["device"], log_threshold = np.log(1))
        
        print("Plotting conditional histograms...")
        if any(len(v) > 0 for model_data in data_for_plotting.values() for v in model_data.values()):
            plot_conditional_histograms(data_for_plotting, seed=seed, bins=100)
            # plot_conditional_histograms_CoTETE(data_for_plotting, seed=seed, bins=100)
        else:
            print("No data was collected for plotting.")

    event_rate = len_target / configs["data_prep_config"]["total_time"]
    TE_hazard *= event_rate
    H_yy_hazard *= event_rate
    H_yyx_hazard *= event_rate
    return (TE_hazard, H_yy_hazard, H_yyx_hazard), (log_loss_yy, log_loss_yyx)


def run_multiple_estimation(target_events, source_events, configs, n_runs=10, seed=42):
    """
    Runs TE estimation multiple times with different random seeds to get variance estimates.
    This mimics run_k_fold_estimation but uses prepare_dataloaders for consistent data splitting.
    """
    print(f"--- Starting {n_runs} Multiple Runs Estimation ---")
    
    # time_series_length = configs["data_prep_config"]["total_time"]
    # len_target = len(target_events)
    
    # Lists to store results from each run
    run_results = []

    # Run multiple estimations with different seeds
    for run in tqdm(range(n_runs)):
        print(f"\n--- Run {run+1}/{n_runs} ---")
        run_start_time = time.time()
        
        # Use different seed for each run to get different train/val/test splits
        run_seed = seed + (run+1) * 1000
        
        # Use the existing TE_estimation_tpp function which handles data preparation
        (TE_sec, H_yy_sec, H_yyx_sec), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
            event_time=[target_events, source_events], 
            configs=configs, 
            seed=run_seed
        )
        
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time
        
        print(f"Run {run+1} Results: TE = {TE_sec:.10f}, h_yy = {H_yy_sec:.3f}, h_yyx = {H_yyx_sec:.3f}")
        print(f"Loss - yy: {log_loss_yy:.3f}, yyx: {log_loss_yyx:.3f}")
        print(f"Run {run+1} completed in {run_duration/60:.2f} minutes.")
        
        # Store results
        run_results.append({
            "TE_sec": TE_sec,
            "h_yy_sec": H_yy_sec,
            "h_yyx_sec": H_yyx_sec,
            "loss_yy": log_loss_yy, 
            "loss_yyx": log_loss_yyx,
            "run_duration_sec": run_duration,
        })

        # save results after each run
        results_df = pd.DataFrame(run_results)
        results_df.to_csv("results/multiple_runs_results.csv", index=False)
    
    # Aggregate and report final results
    results_df = pd.DataFrame(run_results)
    print("\n--- Multiple Runs Summary ---")
    print(results_df[["TE_sec", "h_yy_sec", "h_yyx_sec", "loss_yy", "loss_yyx"]].describe().T)
    
    # The final reported TE is the mean across all runs
    mean_te = results_df['TE_sec'].mean()
    std_te = results_df['TE_sec'].std()
    print(f"\nFinal TE Estimate: {mean_te:.5f} ± {std_te:.5f} nats/second (mean ± std over {n_runs} runs)")
    
    # Save detailed results
    results_df.to_csv("results/multiple_runs_results.csv", index=False)
    
    # Create visualization plots
    # Plot 1: Conditional entropies
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    df_h = results_df[["h_yy_sec", "h_yyx_sec"]]
    # df_h.to_csv("results/h_sec_multiple_runs.csv", index=False)
    df_h.plot(kind='box', figsize=(10, 6), ax=ax1)
    ax1.set_title("Conditional Entropy Estimation Results - Multiple Runs (nats/second)")
    ax1.set_ylabel("Conditional Entropy (nats per second)")
    ax1.grid()
    fig1.savefig("results/h_sec_multiple_runs.png")
    plt.close(fig1)

    # Plot 2: Transfer entropy
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_te = results_df[["TE_sec"]]
    # df_te.to_csv("results/TE_sec_multiple_runs.csv", index=False)
    df_te.plot(kind='box', figsize=(10, 6), ax=ax2)
    ax2.set_title("Transfer Entropy Estimation Results - Multiple Runs (nats/second)")
    ax2.set_ylabel("Transfer Entropy (nats per second)")
    ax2.grid()
    fig2.savefig("results/TE_sec_multiple_runs.png")
    plt.close(fig2)

    # Plot 3: Log losses
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    df_loss = results_df[['loss_yy', 'loss_yyx']]
    # df_loss.to_csv("results/log_loss_multiple_runs.csv", index=False)
    df_loss.plot(kind='box', ax=ax3)
    ax3.set_title("Log Losses for Marginal and Conditional Models - Multiple Runs")
    ax3.set_ylabel("Log Loss")
    ax3.grid()
    fig3.savefig("results/log_loss_multiple_runs.png")
    plt.close(fig3)


    print(f"\nResults saved to results/ directory")
    print(f"Average run duration: {results_df['run_duration_sec'].mean()/60:.2f} minutes")
    print(f"Total wall-clock time: {results_df['run_duration_sec'].sum()/60:.2f} minutes")
    
    return results_df



if __name__ == "__main__":
    # Config
    seed = 31
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_series_length = 60 * 1   # in seconds, Length of the time series
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    configs = {
        "model_config_yyx": {
            "model_name": "yyx",  # Name of the model, could be arbitrary string
            "context_size": 16,  # Size of the context vector (RNN hidden vector)
            "num_mix_components": 1, # Number of components for a mixture model
            "hidden_sizes": [4, 32], # Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of ANN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU", # Activation function to use in the model
        },
        "train_config_yyx": {
            "L2_weight": 0.000827406850573813,          # L2 regularization parameter
            # "L_entropy_weight": 0.0001264625733948674,      # Weight for the entropy regularization term
            # "L_sep_weight": 2.9247292664457833e-05,               # Weight for the separation regularization term
            # "L_scale_weight":  3.3454456195454425e-09,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "model_config_yy": {
            "model_name": "yy",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 16,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 1,#4,  # 16 Number of components for a mixture model
            "hidden_sizes": [32],       # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "train_config_yy": {
            "L2_weight": 0.0001907358691282098,          # L2 regularization parameter
            # "L_entropy_weight": 0.00021180598933158608,      # Weight for the entropy regularization term
            # "L_sep_weight": 1.6394325044339145e-07,               # Weight for the separation regularization term
            # "L_scale_weight": 6.378480299432731e-06,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "data_prep_config":{
            "batch_size": 128,          # Number of sequences in a batch
            "shuffle": False,                 # Whether to shuffle the time series before splitting into train/val/test
            "total_time": time_series_length,              # in second, Total time of the sequences, , truncated at this time if data exceeds this length
            "verbose": False # Whether to print data preparation statistics
        },
        "device": device,
        "verbose": False,  # Whether to print the training statistics
        "plot_histograms": False,  # Whether to plot the conditional histograms
        "plot_pp": False,  # Whether to plot the P-P plot
        "history_length": 32,             # in number of bins, Length of the history to use for the model
    }

    arrival_times_p1, arrival_times_p2 = simulate_mutually_exciting_hawkes(
        mu1=0.1, alpha11=0, beta11=0, alpha12=1.5, beta12=0.8,
        mu2=10, alpha22=0.1, beta22=0.8, alpha21=0.1, beta21=0.8,
        T_end=time_series_length, seed=seed
    )
    torch.manual_seed(seed+1)
    # arrival_times_poi=gen_poission_event_times(lambda_=len(arrival_times_p2)/time_series_length, T=time_series_length)
    print("Number of events in process target:", len(arrival_times_p1))
    print("Number of events in process source:", len(arrival_times_p2))

    (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
            event_time=[arrival_times_p1, arrival_times_p2], 
            configs=configs, 
            seed=seed
    )
    print(f"Estimated Transfer Entropy (nats per event): {TE_test}")
    print(f"Estimated Transfer Entropy (nats per second): {TE_test * len(arrival_times_p1) / time_series_length}")
    print(f"Estimated H(Y_t+1|Y_t) (nats per event): {H_yy_test}")
    print(f"Estimated H(Y_t+1|Y_t) (nats per second): {H_yy_test * len(arrival_times_p1) / time_series_length}")
    print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per event): {H_yyx_test}")
    print(f"Estimated H(Y_t+1|Y_t,X_t) (nats per second): {H_yyx_test * len(arrival_times_p1) / time_series_length}")
    print(f"Log Loss H(Y_t+1|Y_t): {log_loss_yy}")
    print(f"Log Loss H(Y_t+1|Y_t,X_t): {log_loss_yyx}")

    # Run multiple estimations with different seeds for variance assessment
    run_multiple_estimation(
        target_events=arrival_times_p1,
        source_events=arrival_times_p2,
        configs=configs,
        n_runs=20,  # Number of runs with different seeds
        seed=seed
    )
    
    # # Run k-fold cross-validation for robust estimation
    # run_k_fold_estimation(
    #     target_events=arrival_times_p1,
    #     source_events=arrival_times_p2,
    #     configs=configs,
    #     n_splits=5,  # You can choose the number of splits, 5 or 10 are common
    #     seed=seed
    # )
import numpy as np
from entropy_tpp import TE_estimation_tpp
from copy import deepcopy
import torch
import optuna

def create_objective(arrival_times_p1, arrival_times_p2,
                     time_series_length, device, seed):
    """
    This outer function creates and returns the actual objective function.
    It takes the data as an argument.
    """
    
    def objective(trial):
        # 1. Suggest hyperparameters

        context_size = 2** trial.suggest_int("context_size", 0, 3)  # From 2^0 to 2^3, i.e., 1 to 8
        num_mix_components = 2** trial.suggest_int("num_mix_components", 3, 8)

        # Suggest the number of hidden layers
        n_layers = trial.suggest_int("n_layers", 1, 5) # From 1 to 3 hidden layers
        hidden_sizes = []
        for i in range(n_layers):
            # Suggest the size for each hidden layer dynamically
            layer_size = 2** trial.suggest_int(f"hidden_size_l{i}", 0, 8)
            hidden_sizes.append(layer_size)

        rnn_type = trial.suggest_categorical("rnn_type", ["GRU", "LSTM", "RNN"])
        bin_embedding_dim = 2** trial.suggest_int("bin_embedding_dim", 0, 5)
        batch_size = 128
        regularization = 1e-5
        learning_rate = 5e-5 # trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)#1e-4
        max_epochs = 1000
        patience = 20 # Early stopping patience
        history_length = 2** trial.suggest_int("history_length", 0, 8)  # From 2^0 to 2^7, i.e., 1 to 128 bins
        bin_width = 1e-3* 2** trial.suggest_int("bin_width", 0, 8)  # in seconds

        configs = {
            "model_config": {
                "context_size": context_size,                # 8 Size of the RNN hidden vector
                "num_mix_components": num_mix_components,        # 16 Number of components for a mixture model
                "hidden_sizes": hidden_sizes,       # 16 Hidden sizes of the MLP for the inter-event time distribution
                "rnn_type": rnn_type,         # Type of RNN to use, ["GRU", "LSTM", "RNN"]
                "bin_embedding_dim": bin_embedding_dim,        # Dimension of the bin embedding, used for the history embedding
            },
            "train_config": {
                "batch_size": batch_size,                # Number of sequences in a batch
                "regularization": regularization,          # L2 regularization parameter
                "learning_rate": learning_rate,           # Learning rate for Adam optimizer
                "max_epochs": max_epochs,              # For how many epochs to train
                "display_step": 5,               # Display training statistics after every display_step
                "patience": patience,                  # After how many consecutive epochs without improvement of val loss to stop training
                "shuffle": True,                 # Whether to shuffle the time series before splitting into train/val/test
            },
            "device": device,
            "history_length": history_length,             # in number of bins, Length of the history to use for the model
            "bin_width": bin_width,         # in second, Width of the bin for history embedding
            "total_time": time_series_length,              # in second, Total time of the sequences
            "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "num_samples": 10000,  # Number of samples to use for the Monte Carlo sampling in EstimateContinuousTE
            "max_samples_for_qq": 1000,  # Maximum number of samples to use for the Q-Q plot
            "verbose": False,  # Whether to print the training statistics
            "get_final_loss": True,  # Whether to get the final loss on train/val/test sets
        }

        # 2. Run your TE estimation with the suggested hyperparameters
        # You'll need to adapt this part to call the relevant functions in your entropy_tpp.py
        # For instance, if you have a main function that takes these as arguments:
        # te_train, te_val, te_test = calculate_te(param1, param2)
        TEs = []
        TEs_sec = []
        print("Number of events in process 1:", len(arrival_times_p1))
        print("Number of events in process 2:", len(arrival_times_p2))
        len_p1 = len(arrival_times_p1)
        msg = ""
        quantile_yy_losses = []
        quantile_yyx_losses = []
        for i in range(3):
            (TE_train, TE_val, TE_test), (quantile_loss_yy, quantile_loss_yyx), msg1 = TE_estimation_tpp(
                event_time=[arrival_times_p1, arrival_times_p2], 
                configs=deepcopy(configs), 
                seed=seed*(i+1)
            )
            quantile_yy_losses.append(quantile_loss_yy)
            quantile_yyx_losses.append(quantile_loss_yyx)
            msg += f"Run {i+1}:\n{msg1}\n"
            if TE_train == float('nan') or TE_val == float('nan') or TE_test == float('nan'):
                msg += f"Error during TE estimation for run {i+1}. Skipping this run.\n"
                trial.set_user_attr(f"log", msg)
                return None, None
            

            TE_train_sec = TE_train * len_p1 / time_series_length
            TE_val_sec = TE_val * len_p1 / time_series_length
            TE_test_sec = TE_test * len_p1 / time_series_length
            print(f'Transfer Entropy (nats/second) for run {i+1}:\n'
                f' - Train: {TE_train_sec:.5f}\n'
                f' - Val:   {TE_val_sec:.5f}\n'
                f' - Test:  {TE_test_sec:.5f}')
            print(f'Quantile loss for model yy: {quantile_loss_yy:.5f}')
            print(f'Quantile loss for model yyx: {quantile_loss_yyx:.5f}')
            
            trial.set_user_attr(f"TE_train_sec_run_{i}", TE_train_sec)
            trial.set_user_attr(f"TE_val_sec_run_{i}", TE_val_sec)
            trial.set_user_attr(f"TE_test_sec_run_{i}", TE_test_sec)

            trial.set_user_attr(f"quantile_loss_yy_run_{i}", quantile_loss_yy)
            trial.set_user_attr(f"quantile_loss_yyx_run_{i}", quantile_loss_yyx)
            
            TEs.append(TE_test)
            TEs_sec.append(TE_test_sec)
            trial.set_user_attr(f"log", msg)
        
        trial.set_user_attr("TE_sec_test", np.mean(TEs_sec))

        # 3. Return the metric to optimize
        return  np.mean(quantile_yy_losses), np.mean(quantile_yyx_losses)

    return objective


def generate_binned_arrival_times(rate_lambda_per_second: float, bin_width_ms: float, duration_seconds: float, seed = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates two event sequences, X and Y, based on a Poisson process and
    returns the event arrival times.

    Each sequence is generated from a Poisson distribution over fixed-width
    time bins. Sequence Y is a one-bin delayed copy of X. The arrival time
    for each event is calculated as the midpoint of the bin in which it occurred.

    Args:
        rate_lambda_per_second (float): The rate parameter (lambda) of the
                                        Poisson process, in events per second.
        bin_width_ms (float): The width of each time bin in milliseconds.
        duration_seconds (float): The total duration of the sequences in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays of
        event arrival times in seconds for sequence X and Y, respectively.
    """
    # Convert parameters to milliseconds for consistent calculations
    rate_lambda_per_ms = rate_lambda_per_second / 1000.0
    duration_ms = duration_seconds * 1000.0

    # Calculate the total number of bins for the given duration and bin width.
    num_bins = int(duration_ms / bin_width_ms)
    
    # Calculate the average number of events per time bin.
    lambda_per_bin = rate_lambda_per_ms * bin_width_ms

    np.random.seed(seed)
    # Generate the number of events for each bin of sequence X using a Poisson distribution.
    x_events_per_bin = np.random.poisson(lambda_per_bin, num_bins)

    # Convert the event counts to a binary sequence (1 if an event occurred, 0 otherwise).
    x_sequence = np.where(x_events_per_bin > 0, 1, 0)
    
    # Create the y_sequence as a one-bin delayed copy of the x_sequence.
    # The first element is set to 0 as there is no preceding event from X.
    y_sequence = np.roll(x_sequence, 1)
    y_sequence[0] = 0

    # Calculate the arrival times for sequence X.
    # Find the indices where an event (value 1) occurred.
    x_event_indices = np.where(x_sequence == 1)[0]
    # The arrival time is the midpoint of the bin.
    x_arrival_times = (x_event_indices * bin_width_ms) + (bin_width_ms / 2)
    
    # Calculate the arrival times for sequence Y in the same way.
    y_event_indices = np.where(y_sequence == 1)[0]
    y_arrival_times = (y_event_indices * bin_width_ms) + (bin_width_ms / 2)

    return x_arrival_times/1000, y_arrival_times/1000  # Convert to seconds for consistency with the rest of the code

# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters
    # Rate of 10 events per second
    lambda_rate_per_sec = 30
    # Bin width in ms
    bin_width = 1
    # Total duration of 2 seconds
    duration_sec = 60 * 15
    seed = 55

    # Generate the sequences
    x_times, y_times = generate_binned_arrival_times(lambda_rate_per_sec, bin_width, duration_sec, seed=seed)

    print("--- Event Arrival Times for Sequence X ---")
    print(f"Number of events: {len(x_times)}")
    print(x_times[-5:])  # Print the last 5 event times for brevity

    print("--- Event Arrival Times for Sequence Y (one-bin delayed copy) ---")
    print(f"Number of events: {len(y_times)}")
    print(y_times[-5:])  # Print the last 5 event times for brevity

    # It's number of events per bin, when it smaller than 1, it is the probability of an event in a bin.
    p_per_bin = lambda_rate_per_sec  / 1000.0 * bin_width
    print(f"Probability per bin: {p_per_bin}")
    assert p_per_bin <= 1.0, "Probability per bin must be less than or equal to 1."

    diff_entropy_bit = -p_per_bin * np.log2(p_per_bin) - (1 - p_per_bin) * np.log2(1 - p_per_bin)
    diff_entropy_nat = diff_entropy_bit / np.log2(np.e)
    # It's actually X conditioned on Y_history, which is the same as Y in this case.
    # This only happenes when p(Y_present | Y_history) = p(Y_present | X_history) = p(X_present | X_history) = p(X_present)
    # The first equantion holds only when Y is partly dependent on X.

    print(f"\nEntropy for Y conditioned on Y_history: {diff_entropy_bit:.4f} bits per bin")
    # Conditional entropy for Y conditioned on X_history and Y_history is 0 because Y is a delayed copy of X.
    # transfer entropy = diff_entropy_Y_history - diff_entropy_YX_history = diff_entropy_Y_history
    print(f"Transfer entropy from X to Y: {diff_entropy_nat*1000/bin_width :.4f} nats per second")

# --- Optuna Trial ---
    # Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    np.random.seed(seed)
    torch.manual_seed(seed)
    time_series_length = duration_sec   # in seconds, Length of the time series

    objective_t = create_objective(torch.tensor(x_times), torch.tensor(y_times),
                                 time_series_length, device, seed)

    # Assuming 'objective' function is defined as above
    # ,load_if_exists=True to continue from an existing study
    study = optuna.create_study(directions=["minimize","minimize"], storage="sqlite:///db.sqlite3"
                                ,load_if_exists=True, study_name=f"max_te{seed:02d}") # Set direction to 'maximize' for TE,  

    study.optimize(objective_t, n_trials=None) # Run for 100 trials

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")
    
# Documentation

## Summary

This project provides a high-performance pipeline for estimation Transfer Entropy (TE) between continuous-time event sequences (e.g., neural spike trains) using deep learning. This project uses artificial neural networks (Recurrent Mixture Density Networks) to model the conditional density of inter-event intervals (inter-spike intervals), allowing for a robust estimation of TE rate (nats per second) for processes with long temporal dependences.

Additionally, the project offers scripts for HPC parallelization to facilitate high-throughput analysis, including the estimation of transfer entropy between two groups of neurons.

## Theoretical Foundation

Transfer entropy quantifies the amount of predictive information the history of the source process (X) could provide about the current state of the target process (Y).

In the thesis, Tuoxing derived the formula for transfer entropy rate expressed as the difference between two log-density ratio of target inter-event intervals (IEIs). 

$$\begin{equation}
    \begin{split}
    \dot{TE}_{X\to Y}(k,l) &:= \lim_{t-t_0 \to \infty} \frac{1}{t-t_0} \sum^{N_Y([t_0,t))}_{i=1} 
    \left[ - \text{ln} \frac{p_{Y}\left(\tau_Y(i) \mid  \tau_Y^{(k)}(\leq i) \right)}{\int_{\tau_Y(i+1)}^\infty p_{Y}\left(u \mid  \tau_Y^{(k)}(\leq i) \right) \text{d}u} \right.  \\
    &\left. +\text{ln} \frac{p_{Y}\left( \tau_Y(i+1) \mid  \tau_Y^{(k)}(\leq i),\tau_X^{(l)}(\leq N_X(y_i)),y_i-x_{N_X(y_i)} \right)}{\int_{\tau_Y(i+1)}^\infty p_{Y} \left( u \mid  \tau_Y^{(k)}(\leq i),\tau_X^{(l)}(\leq N_X(y_i)),y_i-x_{N_X(y_i)}  \right) \text{d}u} \right]
\end{split}
\end{equation}$$

The TE estimation problem is then converted to following two prediction problem:

1. **(Reduced Model)** Use the historical inter-event intervals of the source to predict the PDF of next inter-event interval 
2. **(Full Model)** Use the historical IEIs of the source and the historical IEIs of the target to predict the PDF of next inter-event interval

Thus we need to train two RMDNs (Recurrent Mixture Density Networks) to model the conditional densities of inter-event intervals **separately** for those two prediction tasks. After training the model, we evaluate them on the ground truth intervals to access its probability. Finally, we can calculate the transfer entropy rate through those probabilities according to the above formula by averaging the logarithm over all events in the target process.

The `k,l` in the formula is the parameter of this method, which specifies the number of historical IEIs in consideration. Theoretically, the larger the `k,l`, the more accurate the estimation. However, due to the increase of dimensionality in the condition, the amount of data needed for the accurate estimation would also increase.

## project Architecture

`data/`: Directory intended for input datasets.

`demo/`: Contains the demonstration notebook (`demo.ipynb`) outlining the estimation pipeline for a single local run.

`dpp/`: Contains a dependent module related to the deep point process models.

`src/`: Primary source code for the transfer entropy estimation framework.

- **`src/te_tpp.py`**: The core source file containing the pipeline for single estimation, including data preprocessing, training of `LogNormMix` models, and the implementation of estimation algorithm via `Estimate_TE_Hazard` function.
- `src/CoTETE*.py`: Scripts to generate synthetic data and compare this program with the leading non-parametric TE estimation method (CoTETE https://github.com/dpshorten/CoTETE.jl).
- `src/plot*.py`: Scripts to generate the plots in the thesis.

`hpc/`: Source code for scaling the estimation framework on high-performance computing (HPC) clusters

- `hpc_header.py`: Including functions needed for other files and function to generate task lists `tasks_reduced.csv` and `tasks_full.csv`, which specifies the (pairs of) processes needs to be processed in the input HDF5 (Hierarchical Data Format 5) file.
- `opt_reduced_model_batch.sh`: SLURM (Simple Linux Utility for Resource Management) script to run the hyperparameter optimization on the cluster for the reduced model, calling the `hyper_opt_reduced.py` file.
- `opt_reduced_model_batch.sh`: SLURM script to run the hyperparameter optimization on the cluster for the full model, calling the `hyper_opt_full.py` file.
- `run_reduced_model_batch.sh`: SLURM script to run multiple estimations of the log ratio with the reduced model, calling the `multi_runs_reduced.py` file.
- `run_full_model_batch.sh`: SLURM script to run multiple estimations of the log ratio with the full model, calling the `multi_runs_full.py` file.
- `aggregate_results.py`: Aggregate estimation results on HPC into a single HDF5 file.
- `calculate_te.py`: Calculate the transfer entropy rate using the generated HDF5 file, and also calculate some statistics of the mean estimation like lower and upper bounds of 95% confidence intervals.
- `plot_te_heatmap.py`: Visualize the TE estimation results between two groups of neuron.

`results/`: Directory for storing computed outputs or plots.

`setup.py`: Package configuration file for managing dependencies.

`requirements.txt & environment.yml`: Listing Python packages required for this project.

## Main Functions in `src/te_tpp.py`

### `TE_estimation_tpp`

The core function which implements a single transfer entropy rate estimation.

```
TE_estimation_tpp(event_time, configs: dict, seed: int = 42, trial=None):
    """
    Estimate the transfer entropy (TE) between two temporal point processes (TPPs) using neural models.
    This function prepares data loaders, trains two TPP models (one with and one without access to the source process),
    and estimates the transfer entropy by comparing the log-probabilities of inter-event times under both models.
    Inputs:
        event_time: A list of torch.Tensor. Input event time data for the processes in seconds. It should be a list of event times for multiple processes. The first tensor in the list is considered the target process, and the second one is the source process.
        configs (dict): Configuration dictionary containing model and training parameters.
        seed (optional): An integer, the random seed for reproducibility.
    Outputs:
        tuple:
            - (TE_test, ln_yy_test, ln_yyx_test): Estimated transfer entropy values per second for the train, validation, and test sets.
            - (log_loss_yy, log_loss_yyx): Quantile losses for the two trained models.
    Raises:
        Returns NaN values and an error message in case of exceptions during model training or TE estimation.
    
    """
```

**Inputs**

- `event_time` should be a list of `torch.Tensor` arrays as input, which contains the absolute event times for the processes. For example, when there are events happens at 1, 3, 5 seconds of the source process, and at 2, 4, 6 seconds for the target process, the input should be:

```
[torch.tensor([2,4,6]),torch.tensor([1,3,5])]
```

- `configs` contains the hyperparameters listed here, with description in the comments.
```
    configs = {
            "model_config_yyx": {
                "model_name": "yyx", # Name of the model, could be arbitrary string
                "context_size": 16, # Size of the context vector (RNN hidden vector)
                "num_mix_components": 1, # Number of components for a mixture model
                "hidden_sizes": [4, 32], # Hidden sizes of the MLP for the inter-event interval distribution
                "context_extractor": "gru", # Type of ANN to use for context extraction, ["gru", "lstm", "mlp"]
                "activation_func": "GELU", # Activation function to use in the model
            },
            "train_config_yyx": {
                "L2_weight": 1e-4, # L2 regularization parameter
                "L_entropy_weight": 0, # Weight for the entropy regularization term
                "L_sep_weight": 0, # Weight for the separation regularization term
                "L_scale_weight":  1e-4, # Weight for the scale regularization term
                "learning_rate": 5e-4, # Learning rate for Adam optimizer
                "max_epochs": 1000, # For how many epochs to train
                "display_step": 5, # Display training statistics after every display_step
                "patience": 40, # After how many consecutive epochs without improvement of validation loss to stop training
            },
            "model_config_yy": {
                "model_name": "yy",
                ...
            },
            "train_config_yy": {
                ...
            },
            "data_prep_config":{
                "batch_size": 128, # Number of sequences in one batch
                "shuffle": False, # Whether to shuffle the source to break its relationship with the target
                "total_time": 300,# In second, total time of the whole sequence(s), truncated at this time if data exceeds this length
                "verbose": False # Whether to print data preparation statistics
            },
            "device": "cuda", # device used for the deep learning
            "verbose": False,  # Whether to print the training statistics
            "plot_histograms": False,  # Whether to plot the conditional histograms
            "plot_pp": False,  # Whether to plot the P-P plot
            "history_length": 32, # In number of inter-event intervals, length of the history to use for the model
        }
```

​    By default, this framework use the same length for the history of the source and target. $k=l=$`history_length`. Later we will use program to automatically choose the best hyperparameters for accurate estimation.

- `seed` sets the random seed for the partition of the dataset and the training process. Using the same seed should get almost the same results.

**Outputs**

- `(TE_test, ln_yy_test, ln_yyx_test)`:  `TE_test` gives the estimation of transfer entropy rate in nats per second. `ln_yy_test` is the log ratio for the reduced model on the test dataset. `ln_yyx_test` is the log ratio for the full model on the test set. `TE_test=ln_yyx_test-ln_yy_test`.
- `(log_loss_yy, log_loss_yyx)`:  The negative log likelihood loss of the reduced and full model on the test dataset.

### `Ln_estimation_yy` and `Ln_estimation_yyx`

These two functions handle specific sub-tasks within the `TE_estimation_tpp` routine. They train the reduced and full model respectively and estimate the corresponding log ratio in the formula. They take similar inputs as `TE_estimation_tpp`, and returns `ln_yy_test, ln_yyx_test`, where `TE_test=ln_yyx_test-ln_yy_test` and their negative log likelihood loss on the test dataset.

## Workflow Pipeline (Local)

input: event time array

## Workflow Pipeline (HPC)

accept h5

## Metrics for Evaluation

## Requirements
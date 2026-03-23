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

### `Ln_estimation_yyx` and `Ln_estimation_yyx`

## Workflow Pipeline (Local)

input: event time array

## Workflow Pipeline (HPC)

accept h5

## Metrics for Evaluation

## Requirements
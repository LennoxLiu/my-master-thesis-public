# Continuous-Time Transfer Entropy Estimation for Point Processes via Recurrent Mixture Density Networks: Application to Spike Trains

This project provides a deep learning pipeline to estimate **Transfer Entropy (TE) rate** between continuous-time event sequences (e.g., neural spike trains). It uses **Recurrent Mixture Density Networks (RMDNs)** to model the conditional density of inter-event intervals, enabling robust TE estimation (in nats/sec) for processes with long temporal dependencies.

The project also includes scripts for **HPC parallelization** (SLURM), supporting high-throughput pairwise TE estimation across large neuron populations.

Modified from the PyTorch implementation of [*Intensity-Free Learning of Temporal Point Processes*](https://openreview.net/forum?id=HygOjhEYDH) (Shchur, Biloš & Günnemann, ICLR 2020).

> **Full documentation:** see [`documentation/Documentation.md`](./documentation/Documentation.md)

------

## How It Works

TE is estimated by training two separate RMDNs:

1. **Reduced model** — predicts the next inter-event interval (IEI) of the target process using only its own history.
2. **Full model** — predicts the next target IEI using both the target's and the source's history.

The TE rate is then computed as:

$$\dot{TE} = \text{mean}(\ln_{yyx}) - \text{mean}(\ln_{yy})$$

where $\ln_{yy}$ and $\ln_{yyx}$ are the per-event log-ratios from the reduced and full models respectively.

------

## Quick Start (Local)

**1. Install**

```bash
conda env create -f environment.yml
conda activate RMDN-TE
pip install -e .
```

**2. Run the demo**

Open `demo/demo.ipynb` for a complete worked example of the estimation pipeline on a single neuron pair.

**3. Run programmatically**

```python
import torch
from src.te_tpp import TE_estimation_tpp

event_time = [
    torch.tensor([2., 4., 6., ...]),  # Target process (first)
    torch.tensor([1., 3., 5., ...]),  # Source process (second)
]

configs = { ... }  # See Documentation.md for full config reference

(TE_test, ln_yy_test, ln_yyx_test), (log_loss_yy, log_loss_yyx) = \
    TE_estimation_tpp(event_time, configs, seed=42)

print(f"TE rate: {TE_test:.4f} nats/sec")
```

------

## HPC Pipeline (SLURM)

For large-scale pairwise estimation across neuron populations, the full pipeline is:

```
Input HDF5
    │
    ├──► [Step 0]  Generate task lists       python hpc/hpc_header.py
    ├──► [Step 1]  Hyperparameter search     sbatch hpc/opt_{reduced,full}_model_batch.sh
    ├──► [Step 2]  Multi-run estimation      sbatch hpc/run_{reduced,full}_model_batch.sh
    ├──► [Step 3]  Aggregate results         python hpc/aggregate_results.py
    ├──► [Step 4]  Calculate TE              python hpc/calculate_te.py
    └──► [Step 5]  Visualize                 python hpc/plot_te_heatmap.py
```

If your data is in MATLAB format, first convert it to HDF5 format. For example,

```bash
python hpc/mat_to_h5.py  # converts data/testFile.mat → data/event_times_data.h5
```

See [`documentation/Documentation.md`](./documentation/Documentation.md) for the full step-by-step HPC guide, including SLURM parameters, input/output formats, and progress checking.

------

## Project Structure

```
.
├── data/             # Input datasets (HDF5 and MAT)
├── demo/             # Demonstration notebook and results
├── documentation/    # Full documentation
├── dpp/              # Deep point process model utilities
├── hpc/              # HPC parallelization scripts (SLURM)
├── results/          # Computed outputs and plots
├── src/              # Core TE estimation source code
├── environment.yml   # Conda environment
└── setup.py          # Package configuration
```

------

## Requirements

| Package      | Version                                  |
| ------------ | ---------------------------------------- |
| Python       | 3.12                                     |
| PyTorch      | 2.5.1                                    |
| pytorch-cuda | 12.4 (optional)                          |
| NumPy        | 2.2.5                                    |
| SciPy        | 1.15.3                                   |
| Optuna       | 4.4.0                                    |
| pandas       | 2.2.3                                    |
| h5py         | 3.15.1                                   |
| scikit-learn | 1.6.1                                    |
| juliacall    | 0.9.28 (optional, for CoTETE comparison) |

On **bwUniCluster 2.0 / bwForCluster**, load the required modules first:

```bash
module load devel/miniforge/24.9.2
module load devel/cuda/12.6
```

------

## License

See [LICENSE.md](./LICENSE.md) for details.
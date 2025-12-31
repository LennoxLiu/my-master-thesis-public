import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from entropy_tpp import TE_estimation_tpp, run_multiple_estimation
import torch
from numpy.polynomial.laguerre import laggauss
import time
from scipy.stats import lognorm
from numpy.polynomial.hermite import hermgauss
import math
from entropy_tpp import save_dict_indented
from contextlib import redirect_stdout
import pandas as pd
from piecewise_lognormal import simulate_processes, compute_reference


def plot_conditional_histograms(plot_data: dict, seed: int, bins: int = 100):
    fig, axes = plt.subplots(4, 1, figsize=(8, 22), sharex=False, sharey=False)
    fig.suptitle("Simulated data", fontsize=16)

    conditions= {
        (True, True): "SS",
        (True, False): "SL",
        (False, True): "LS",
        (False, False): "LL"
    }

    x_maxes = { "LL": 0.05, "LS": 0.2, "SL": 0.3, "SS": 0.5 }
    for ax, (key, values) in zip(axes, plot_data.items()):
        ax.set_xlim(0, x_maxes[conditions[key]])
        x_values = np.array(values)
        ax.hist(x_values[x_values<x_maxes[conditions[key]]], bins=bins, color='lightgreen', alpha=0.7,density=True,label=f"Samples: {len(x_values)}, Max: {np.max(x_values):.4f}")
        ax.set_ylabel(conditions[key])
        ax.grid(True)
        ax.legend()
        # print(f"Condition {conditions[key]}: max={np.max(values):.4f}")


    axes[-1].set_xlabel('Inter-event Time (seconds)', fontsize=14)
    fig.text(0.02, 0.5, 'Counts', va='center', rotation='vertical', fontsize=14)


    os.makedirs(f"./results/hists/", exist_ok=True)
    # Delete all files in the directory before saving
    hist_dir = "./results/hists/"
    if os.path.exists(hist_dir):
        for filename in os.listdir(hist_dir):
            file_path = os.path.join(hist_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    
    plt.savefig(f"./results/hists/cond_hist_simu_{seed}.png", dpi=300)
    # plt.show()


def create_morphed_intensity_table(morph_factor):
    """
    Create an intensity table with morphed parameters.
    
    Args:
        morph_factor (float): Value between 0 and 1 where:
            - 0: Original distinct parameters
            - 1: All parameters converged (no transfer entropy)
    
    Returns:
        dict: Morphed intensity table
    """
    # Original parameters (distinct)
    original_table = {
        (False, False): (-5, 0.5),  # LL
        (False, True): (-7, 2),     # LS  
        (True, False): (-3, 0.5),   # SL
        (True, True): (-4, 1.5)     # SS
    }
    
    # Target parameters (converged)
    # Make (False, False) close to (False, True)
    # Make (True, False) close to (True, True)
    target_table = {
        (False, False): (-7, 2),     # LL -> LS
        (False, True): (-7, 2),      # LS (unchanged)
        (True, False): (-4, 1.5),    # SL -> SS  
        (True, True): (-4, 1.5)      # SS (unchanged)
    }
    
    # Interpolate between original and target
    morphed_table = {}
    for key in original_table:
        orig_mu, orig_sigma = original_table[key]
        target_mu, target_sigma = target_table[key]
        
        # Linear interpolation
        morphed_mu = orig_mu + morph_factor * (target_mu - orig_mu)
        morphed_sigma = orig_sigma + morph_factor * (target_sigma - orig_sigma)
        
        morphed_table[key] = (morphed_mu, morphed_sigma)
    
    return morphed_table


def save_incremental_results(results, step_idx, save_prefix="morphing_incremental"):
    """
    Save results incrementally after each morphing step by appending to existing files.
    
    Args:
        results (dict): Current results dictionary
        step_idx (int): Current step index (0-based)
        save_prefix (str): Prefix for saved files
    """
    import pandas as pd
    
    # Create incremental results directory
    incremental_dir = "results/incremental"
    os.makedirs(incremental_dir, exist_ok=True)
    
    # Get current step data
    current_factor = results['morph_factors'][step_idx]
    current_te_runs = results['te_all_runs'][step_idx]
    current_h_yy_runs = results['h_yy_all_runs'][step_idx]
    current_h_yyx_runs = results['h_yyx_all_runs'][step_idx]
    current_intensity_table = results['intensity_tables'][step_idx]
    current_sim_stats = results['simulation_stats'][step_idx]
    current_step_runtime = results['step_runtimes'][step_idx]
    current_estimation_runtimes = results['estimation_runtimes'][step_idx]
    current_log_loss_yy_runs = results['log_loss_yy_all_runs'][step_idx]
    current_log_loss_yyx_runs = results['log_loss_yyx_all_runs'][step_idx]
    current_te_ref = results['te_ref'][step_idx]
    current_h_yy_ref = results['h_yy_ref'][step_idx]
    current_h_yyx_ref = results['h_yyx_ref'][step_idx]

    # File paths
    individual_runs_file = f'{incremental_dir}/{save_prefix}_individual_runs.csv'
    summary_file = f'{incremental_dir}/{save_prefix}_summary.csv'
    intensity_file = f'{incremental_dir}/{save_prefix}_intensity_tables.csv'
    sim_stats_file = f'{incremental_dir}/{save_prefix}_simulation_stats.csv'
    runtimes_file = f'{incremental_dir}/{save_prefix}_runtimes.csv'
    
    # Save individual runs for current step
    
    # For now, skip reference calculation to avoid errors during testing
    # TODO: Add proper reference calculation with correct simulation_time parameter
    
    detailed_data = []
    for run_idx in range(len(current_te_runs)):
        detailed_data.append({
            'morph_factor': current_factor,
            'run_number': run_idx + 1,
            'te_value': current_te_runs[run_idx],
            'te_ref': current_te_ref[run_idx],
            'h_yy_value': current_h_yy_runs[run_idx],
            'h_yyx_value': current_h_yyx_runs[run_idx],
            'h_yy_ref': current_h_yy_ref[run_idx],
            'h_yyx_ref': current_h_yyx_ref[run_idx],
            'estimation_runtime': current_estimation_runtimes[run_idx],
            'log_loss_yy': current_log_loss_yy_runs[run_idx],
            'log_loss_yyx': current_log_loss_yyx_runs[run_idx],
        })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Append to individual runs file (create if first step)
    if step_idx == 0:
        detailed_df.to_csv(individual_runs_file, index=False)
    else:
        detailed_df.to_csv(individual_runs_file, mode='a', header=False, index=False)
    
    # Save summary statistics for current step
    summary_data = [{
        'morph_factor': current_factor,
        'te_mean': np.mean(current_te_runs),
        'te_ref': np.mean(current_te_ref),
        'te_std': np.std(current_te_runs),
        'h_yy_mean': np.mean(current_h_yy_runs),
        'h_yy_ref': np.mean(current_h_yy_ref),
        'h_yy_std': np.std(current_h_yy_runs),
        'h_yyx_mean': np.mean(current_h_yyx_runs),
        'h_yyx_ref': np.mean(current_h_yyx_ref),
        'h_yyx_std': np.std(current_h_yyx_runs),
        'step_runtime': current_step_runtime,
        'avg_estimation_runtime': np.mean(current_estimation_runtimes),
        'total_estimation_runtime': np.sum(current_estimation_runtimes),
        'log_loss_yy_mean': np.mean(current_log_loss_yy_runs),
        'log_loss_yy_std': np.std(current_log_loss_yy_runs),
        'log_loss_yyx_mean': np.mean(current_log_loss_yyx_runs),
        'log_loss_yyx_std': np.std(current_log_loss_yyx_runs)
    }]
    
    summary_df = pd.DataFrame(summary_data)
    
    # Append to summary file (create if first step)
    if step_idx == 0:
        summary_df.to_csv(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)
    
    # Save intensity tables for current step
    intensity_data = []
    for condition, (mu, sigma) in current_intensity_table.items():
        condition_name = {
            (False, False): "LL", (False, True): "LS",
            (True, False): "SL", (True, True): "SS"
        }[condition]
        intensity_data.append({
            'morph_factor': current_factor,
            'condition': condition_name,
            'mu': mu,
            'sigma': sigma
        })
    
    intensity_df = pd.DataFrame(intensity_data)
    
    # Append to intensity file (create if first step)
    if step_idx == 0:
        intensity_df.to_csv(intensity_file, index=False)
    else:
        intensity_df.to_csv(intensity_file, mode='a', header=False, index=False)
    
    # Save simulation statistics for current step
    sim_stats_data = [{
        'morph_factor': current_factor,
        'n_x_events': current_sim_stats['n_x_events'],
        'n_y_events': current_sim_stats['n_y_events'],
        'rate_x': current_sim_stats['rate_x'],
        'rate_y': current_sim_stats['rate_y']
    }]
    
    sim_stats_df = pd.DataFrame(sim_stats_data)
    
    # Append to simulation stats file (create if first step)
    if step_idx == 0:
        sim_stats_df.to_csv(sim_stats_file, index=False)
    else:
        sim_stats_df.to_csv(sim_stats_file, mode='a', header=False, index=False)
    
    # Save runtime data for current step
    runtime_data = []
    for run_idx in range(len(current_estimation_runtimes)):
        runtime_data.append({
            'morph_factor': current_factor,
            'run_number': run_idx + 1,
            'estimation_runtime': current_estimation_runtimes[run_idx]
        })
    
    # Add step summary runtime info
    runtime_data.append({
        'morph_factor': current_factor,
        'run_number': 'step_total',
        'estimation_runtime': current_step_runtime
    })
    
    runtime_df = pd.DataFrame(runtime_data)
    
    # Append to runtime file (create if first step)
    if step_idx == 0:
        runtime_df.to_csv(runtimes_file, index=False)
    else:
        runtime_df.to_csv(runtimes_file, mode='a', header=False, index=False)
    
    print(f"  → Step {step_idx + 1} results appended to incremental files in {incremental_dir}/")


def run_morphing_test(
    morph_factors: list, 
    configs: dict,
    simulation_time: float = 15*60,
    lambda_x: float = 30,
    n_estimations: int = 3,
    base_seed: int = 64
):
    """
    Run the morphing test across different parameter combinations.
    
    Args:
        n_morph_steps (int): Number of morphing steps (including 0 and 1)
        simulation_time (float): Simulation duration in seconds
        lambda_x (float): Poisson process rate
        n_estimations (int): Number of TE estimations per morphing step
        base_seed (int): Base random seed
    
    Returns:
        dict: Results dictionary with morphing factors and corresponding TEs
    """
    print(f"=== Starting Morphing Test ===")
    
    # Create morphing factors from 0 to 1
    morph_factors = sorted(set(morph_factors))  # Remove duplicates and sort
    assert all(0 <= mf <= 1 for mf in morph_factors), "Morph factors must be between 0 and 1"
    n_morph_steps = len(morph_factors)  # Update in case of rounding
    
    results = {
        'morph_factors': [],
        'te_all_runs': [],  # Store all individual TE values
        'te_ref': [],
        'h_yy_ref': [],
        'h_yyx_ref': [],
        'h_yy_all_runs': [],  # Store all individual H_yy values
        'h_yyx_all_runs': [],  # Store all individual H_yyx values
        'intensity_tables': [],
        'simulation_stats': [],
        'step_runtimes': [],  # Store total runtime for each morphing step
        'estimation_runtimes': [],  # Store individual estimation runtimes
        'log_loss_yy_all_runs': [],  # Store all individual log loss Y|Y values
        'log_loss_yyx_all_runs': []  # Store all individual log loss Y|Y,X values
    }

    for i, morph_factor in tqdm(enumerate(morph_factors)):
        step_start_time = time.time()  # Start timing this morphing step
        
        print(f"\n{'='*60}")
        print(f"Morphing Step {i+1}/{n_morph_steps}: factor = {morph_factor:.2f}")
        print(f"{'='*60}")
        
        # Create morphed intensity table
        intensity_table = create_morphed_intensity_table(morph_factor)
        print("Intensity table:")
        for key, (mu, sigma) in intensity_table.items():
            condition_name = {
                (False, False): "LL", (False, True): "LS",
                (True, False): "SL", (True, True): "SS"
            }[key]
            print(f"  {condition_name}: μ={mu:.3f}, σ={sigma:.3f}")
        
        # Simulate data with current parameters
        step_seed = base_seed + i * 100
        np.random.seed(step_seed)
        torch.manual_seed(step_seed)
        
        print(f"Simulating data with seed {step_seed}...")
        x_events, y_events, count_table, plot_data = simulate_processes(
            simulation_time, lambda_x, intensity_table, step_seed
        )
        
        sim_stats = {
            'n_x_events': len(x_events),
            'n_y_events': len(y_events),
            'rate_x': len(x_events) / simulation_time,
            'rate_y': len(y_events) / simulation_time,
            'count_table': count_table
        }
        
        print(f"Generated {len(y_events)} Y events and {len(x_events)} X events")
        print(f"Y rate: {sim_stats['rate_y']:.2f} events/sec")
        
        h_yy_ref, h_yyx_ref, te_ref = compute_reference(
            count_table, intensity_table, simulation_time
        )

        # Run multiple TE estimations
        print(f"Running {n_estimations} TE estimations...")
        
        te_values = []
        h_yy_values = []
        h_yyx_values = []
        estimation_times = []  # Track individual estimation runtimes
        log_loss_yy_values = []  # Track log loss Y|Y values
        log_loss_yyx_values = []  # Track log loss Y|Y,X values
        
        for est_run in range(n_estimations):
            est_start_time = time.time()  # Start timing this estimation
            est_seed = step_seed + est_run + 1
            print(f"  Estimation {est_run+1}/{n_estimations} (seed: {est_seed})")
            
            (TE_test, H_yy_test, H_yyx_test), (log_loss_yy, log_loss_yyx) = TE_estimation_tpp(
                event_time=[torch.tensor(y_events), torch.tensor(x_events)], 
                configs=configs, 
                seed=est_seed
            )
            
            est_end_time = time.time()
            est_runtime = est_end_time - est_start_time
            estimation_times.append(est_runtime)
            
            # Convert to per-second rates
            te_sec = TE_test * len(y_events) / simulation_time
            h_yy_sec = H_yy_test * len(y_events) / simulation_time
            h_yyx_sec = H_yyx_test * len(y_events) / simulation_time
            
            te_values.append(te_sec)
            h_yy_values.append(h_yy_sec)
            h_yyx_values.append(h_yyx_sec)
            log_loss_yy_values.append(log_loss_yy)
            log_loss_yyx_values.append(log_loss_yyx)
            
            print(f"    TE_sec: {te_sec:.4f}, h_yy_sec: {h_yy_sec:.4f}, h_yyx_sec: {h_yyx_sec:.4f}")
            print(f"    log_loss_yy: {log_loss_yy:.4f}, log_loss_yyx: {log_loss_yyx:.4f} (runtime: {est_runtime/60:.2f} mins)")
        
        # Calculate statistics for display
        te_mean = np.mean(te_values)
        te_std = np.std(te_values)
        
        step_end_time = time.time()
        step_runtime = step_end_time - step_start_time
        
        print(f"Results: TE = {te_mean:.4f} ± {te_std:.4f} nats/sec")
        print(f"Step runtime: {step_runtime:.2f}s (avg per estimation: {np.mean(estimation_times):.2f}s)")
        
        # Store results (only individual values, calculate means/stds when needed)
        results['morph_factors'].append(morph_factor)
        results['te_all_runs'].append(te_values.copy())  # Store all individual TE values
        results['h_yy_all_runs'].append(h_yy_values.copy())  # Store all individual H_yy values
        results['h_yyx_all_runs'].append(h_yyx_values.copy())  # Store all individual H_yyx values
        results['intensity_tables'].append(intensity_table)
        results['simulation_stats'].append(sim_stats)
        results['step_runtimes'].append(step_runtime)
        results['estimation_runtimes'].append(estimation_times.copy())
        results['log_loss_yy_all_runs'].append(log_loss_yy_values.copy())
        results['log_loss_yyx_all_runs'].append(log_loss_yyx_values.copy())
        results['te_ref'].append([te_ref for _ in range(len(te_values))])  # Repeat ref for each run
        results['h_yy_ref'].append([h_yy_ref for _ in range(len(h_yy_values))])  # Repeat ref for each run
        results['h_yyx_ref'].append([h_yyx_ref for _ in range(len(h_yyx_values))])  # Repeat ref for each run

        # Save incremental results after each step
        save_incremental_results(results, i, save_prefix="morphing_test")
    
    return results


def plot_morphing_results(results, save_prefix="morphing"):
    """
    Plot the morphing test results in separate subplots with reference values.
    """
    morph_factors = np.array(results['morph_factors'])
    
    # Calculate means and stds on demand
    te_means = np.array([np.mean(te_runs) for te_runs in results['te_all_runs']])
    te_stds = np.array([np.std(te_runs) for te_runs in results['te_all_runs']])
    h_yy_means = np.array([np.mean(h_runs) for h_runs in results['h_yy_all_runs']])
    h_yy_stds = np.array([np.std(h_runs) for h_runs in results['h_yy_all_runs']])
    h_yyx_means = np.array([np.mean(h_runs) for h_runs in results['h_yyx_all_runs']])
    h_yyx_stds = np.array([np.std(h_runs) for h_runs in results['h_yyx_all_runs']])
    log_loss_yy_means = np.array([np.mean(ll_runs) for ll_runs in results['log_loss_yy_all_runs']])
    log_loss_yy_stds = np.array([np.std(ll_runs) for ll_runs in results['log_loss_yy_all_runs']])
    log_loss_yyx_means = np.array([np.mean(ll_runs) for ll_runs in results['log_loss_yyx_all_runs']])
    log_loss_yyx_stds = np.array([np.std(ll_runs) for ll_runs in results['log_loss_yyx_all_runs']])
    
    # Calculate reference values for each step
    print("Computing reference values...")
    te_refs = []
    h_yy_refs = []
    h_yyx_refs = []
    
    for i, intensity_table in enumerate(results['intensity_tables']):
        count_table = results['simulation_stats'][i]['count_table']
        total_duration = SIMULATION_TIME  # Should match SIMULATION_TIME from main

        h_yy_ref, h_yyx_ref, te_ref = compute_reference(
            count_table, intensity_table, total_duration
        )
        
        te_refs.append(te_ref)
        h_yy_refs.append(h_yy_ref)
        h_yyx_refs.append(h_yyx_ref)
    
    te_refs = np.array(te_refs)
    h_yy_refs = np.array(h_yy_refs)
    h_yyx_refs = np.array(h_yyx_refs)
    
    # Plot 1: Transfer Entropy vs Morphing Factor
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(morph_factors, te_means, yerr=te_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5, color='blue', label='Estimated TE')
    
    # Add reference line
    ax1.plot(morph_factors, te_refs, 'r--', linewidth=2, markersize=6, marker='s', 
             label='Reference TE (TSQ)', alpha=0.8)
    
    # Plot individual runs as scatter points
    for i, (factor, te_runs) in enumerate(zip(morph_factors, results['te_all_runs'])):
        ax1.scatter([factor] * len(te_runs), te_runs, alpha=0.4, s=20, color='lightblue')
    
    ax1.set_xlabel('Morphing Factor', fontsize=12)
    ax1.set_ylabel('Transfer Entropy (nats/sec)', fontsize=12)
    ax1.set_title('Transfer Entropy vs Morphing Factor', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_te_evolution.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Plot 2: Conditional Entropies
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(morph_factors, h_yy_means, yerr=h_yy_stds, 
                marker='s', label='h_yy Estimated', linewidth=2, markersize=8, capsize=5, color='green')
    ax2.errorbar(morph_factors, h_yyx_means, yerr=h_yyx_stds, 
                marker='^', label='h_yyx Estimated', linewidth=2, markersize=8, capsize=5, color='orange')
    
    # Add reference lines
    ax2.plot(morph_factors, h_yy_refs, 'g--', linewidth=2, marker='s', markersize=6,
             label='h_yy Reference (TSQ)', alpha=0.8)
    ax2.plot(morph_factors, h_yyx_refs, color='darkorange', linestyle='--', linewidth=2, 
             marker='^', markersize=6, label='h_yyx Reference (TSQ)', alpha=0.8)

    # Plot individual runs as scatter points
    for i, (factor, hyy_runs, hyyx_runs) in enumerate(zip(morph_factors, results['h_yy_all_runs'], results['h_yyx_all_runs'])):
        ax2.scatter([factor] * len(hyy_runs), hyy_runs, alpha=0.4, s=15, color='lightgreen')
        ax2.scatter([factor] * len(hyyx_runs), hyyx_runs, alpha=0.4, s=15, color='wheat')
    
    ax2.set_xlabel('Morphing Factor', fontsize=12)
    ax2.set_ylabel('Conditional Entropy (nats/sec)', fontsize=12)
    ax2.set_title('Conditional Entropies vs Morphing Factor', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_conditional_entropies.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Plot 3: TE in morphing space with individual points
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    scatter = ax3.scatter(morph_factors, te_means, c=te_means, 
                         s=150, cmap='viridis', alpha=0.8, edgecolors='black')
    
    # Add individual run points
    for i, (factor, te_runs) in enumerate(zip(morph_factors, results['te_all_runs'])):
        ax3.scatter([factor] * len(te_runs), te_runs, alpha=0.4, s=30, color='gray')
    
    ax3.set_xlabel('Morphing Factor', fontsize=12)
    ax3.set_ylabel('Transfer Entropy (nats/sec)', fontsize=12)
    ax3.set_title('TE in Morphing Space', fontsize=14)
    plt.colorbar(scatter, ax=ax3, label='TE (nats/sec)')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_te_morphing_space.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # Plot 4: Parameter evolution
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    # Extract parameter values for visualization
    ll_mu = [table[(False, False)][0] for table in results['intensity_tables']]
    ls_mu = [table[(False, True)][0] for table in results['intensity_tables']]
    sl_mu = [table[(True, False)][0] for table in results['intensity_tables']]
    ss_mu = [table[(True, True)][0] for table in results['intensity_tables']]
    
    ax4.plot(morph_factors, ll_mu, 'o-', label='LL (μ)', linewidth=2, markersize=6)
    ax4.plot(morph_factors, ls_mu, 's-', label='LS (μ)', linewidth=2, markersize=6)
    ax4.plot(morph_factors, sl_mu, '^-', label='SL (μ)', linewidth=2, markersize=6)
    ax4.plot(morph_factors, ss_mu, 'd-', label='SS (μ)', linewidth=2, markersize=6)
    ax4.set_xlabel('Morphing Factor', fontsize=12)
    ax4.set_ylabel('μ parameter', fontsize=12)
    ax4.set_title('Parameter Evolution', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_parameter_evolution.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Plot 5: Log Loss Evolution
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ax5.errorbar(morph_factors, log_loss_yy_means, yerr=log_loss_yy_stds, 
                marker='s', label='Log Loss Y|Y', linewidth=2, markersize=8, capsize=5, color='purple')
    ax5.errorbar(morph_factors, log_loss_yyx_means, yerr=log_loss_yyx_stds, 
                marker='^', label='Log Loss Y|Y,X', linewidth=2, markersize=8, capsize=5, color='brown')
    
    # Plot individual runs as scatter points
    for i, (factor, ll_yy_runs, ll_yyx_runs) in enumerate(zip(morph_factors, results['log_loss_yy_all_runs'], results['log_loss_yyx_all_runs'])):
        ax5.scatter([factor] * len(ll_yy_runs), ll_yy_runs, alpha=0.4, s=15, color='mediumpurple')
        ax5.scatter([factor] * len(ll_yyx_runs), ll_yyx_runs, alpha=0.4, s=15, color='sandybrown')
    
    ax5.set_xlabel('Morphing Factor', fontsize=12)
    ax5.set_ylabel('Log Loss', fontsize=12)
    ax5.set_title('Model Training Log Loss vs Morphing Factor', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_log_loss_evolution.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Plot 6: Combined overview with all individual runs
    fig6, ((ax1_combined, ax2_combined), (ax3_combined, ax4_combined), (ax5_combined, ax6_combined)) = plt.subplots(3, 2, figsize=(15, 18))
    fig6.suptitle('Transfer Entropy Morphing Test - Complete Results', fontsize=16)
    
    # Subplot 1: TE with individual points and reference
    ax1_combined.errorbar(morph_factors, te_means, yerr=te_stds, 
                         marker='o', linewidth=2, markersize=6, capsize=5, color='blue', label='Estimated TE')
    ax1_combined.plot(morph_factors, te_refs, 'r--', linewidth=2, marker='s', markersize=5,
                     label='Reference TE (TSQ)', alpha=0.8)
    
    for i, (factor, te_runs) in enumerate(zip(morph_factors, results['te_all_runs'])):
        ax1_combined.scatter([factor] * len(te_runs), te_runs, alpha=0.4, s=15, color='lightblue')
    ax1_combined.set_xlabel('Morphing Factor')
    ax1_combined.set_ylabel('Transfer Entropy (nats/sec)')
    ax1_combined.set_title('Transfer Entropy vs Morphing Factor')
    ax1_combined.grid(True, alpha=0.3)
    ax1_combined.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1_combined.legend(fontsize=10)
    
    # Subplot 2: Conditional entropies with individual points and references
    ax2_combined.errorbar(morph_factors, h_yy_means, yerr=h_yy_stds, 
                         marker='s', label='H(Y|Y) Estimated', linewidth=2, markersize=6, capsize=5, color='green')
    ax2_combined.errorbar(morph_factors, h_yyx_means, yerr=h_yyx_stds, 
                         marker='^', label='H(Y|Y,X) Estimated', linewidth=2, markersize=6, capsize=5, color='orange')
    ax2_combined.plot(morph_factors, h_yy_refs, 'g--', linewidth=2, marker='s', markersize=5,
                     label='H(Y|Y) Reference', alpha=0.8)
    ax2_combined.plot(morph_factors, h_yyx_refs, color='darkorange', linestyle='--', linewidth=2, 
                     marker='^', markersize=5, label='H(Y|Y,X) Reference', alpha=0.8)
    
    for i, (factor, hyy_runs, hyyx_runs) in enumerate(zip(morph_factors, results['h_yy_all_runs'], results['h_yyx_all_runs'])):
        ax2_combined.scatter([factor] * len(hyy_runs), hyy_runs, alpha=0.4, s=10, color='lightgreen')
        ax2_combined.scatter([factor] * len(hyyx_runs), hyyx_runs, alpha=0.4, s=10, color='wheat')
    ax2_combined.set_xlabel('Morphing Factor')
    ax2_combined.set_ylabel('Conditional Entropy (nats/sec)')
    ax2_combined.set_title('Conditional Entropies vs Morphing Factor')
    ax2_combined.legend(fontsize=9)
    ax2_combined.grid(True, alpha=0.3)
    
    # Subplot 3: All individual TE runs as lines
    ax3_combined.plot(morph_factors, te_means, 'bo-', linewidth=3, markersize=8, label='Mean TE')
    for run_idx in range(len(results['te_all_runs'][0])):  # Number of runs per step
        te_line = [results['te_all_runs'][step][run_idx] for step in range(len(morph_factors))]
        ax3_combined.plot(morph_factors, te_line, 'o-', alpha=0.6, linewidth=1, markersize=4, label=f'Run {run_idx+1}')
    ax3_combined.set_xlabel('Morphing Factor')
    ax3_combined.set_ylabel('Transfer Entropy (nats/sec)')
    ax3_combined.set_title('Individual TE Estimation Runs')
    ax3_combined.grid(True, alpha=0.3)
    ax3_combined.legend()
    
    # Subplot 4: Variance analysis
    te_vars = [np.var(te_runs) for te_runs in results['te_all_runs']]
    hyy_vars = [np.var(hyy_runs) for hyy_runs in results['h_yy_all_runs']]
    hyyx_vars = [np.var(hyyx_runs) for hyyx_runs in results['h_yyx_all_runs']]
    
    ax4_combined.plot(morph_factors, te_vars, 'o-', label='TE Variance', linewidth=2, markersize=6)
    ax4_combined.plot(morph_factors, hyy_vars, 's-', label='H(Y|Y) Variance', linewidth=2, markersize=6)
    ax4_combined.plot(morph_factors, hyyx_vars, '^-', label='H(Y|Y,X) Variance', linewidth=2, markersize=6)
    ax4_combined.set_xlabel('Morphing Factor')
    ax4_combined.set_ylabel('Variance')
    ax4_combined.set_title('Estimation Variance vs Morphing Factor')
    ax4_combined.legend()
    ax4_combined.grid(True, alpha=0.3)
    
    # Subplot 5: Log Loss Y|Y
    ax5_combined.errorbar(morph_factors, log_loss_yy_means, yerr=log_loss_yy_stds, 
                         marker='s', linewidth=2, markersize=6, capsize=5, color='purple', label='Log Loss Y|Y')
    for i, (factor, ll_yy_runs) in enumerate(zip(morph_factors, results['log_loss_yy_all_runs'])):
        ax5_combined.scatter([factor] * len(ll_yy_runs), ll_yy_runs, alpha=0.4, s=10, color='mediumpurple')
    ax5_combined.set_xlabel('Morphing Factor')
    ax5_combined.set_ylabel('Log Loss Y|Y')
    ax5_combined.set_title('Model Training Log Loss Y|Y')
    ax5_combined.legend()
    ax5_combined.grid(True, alpha=0.3)
    
    # Subplot 6: Log Loss Y|Y,X
    ax6_combined.errorbar(morph_factors, log_loss_yyx_means, yerr=log_loss_yyx_stds, 
                         marker='^', linewidth=2, markersize=6, capsize=5, color='brown', label='Log Loss Y|Y,X')
    for i, (factor, ll_yyx_runs) in enumerate(zip(morph_factors, results['log_loss_yyx_all_runs'])):
        ax6_combined.scatter([factor] * len(ll_yyx_runs), ll_yyx_runs, alpha=0.4, s=10, color='sandybrown')
    ax6_combined.set_xlabel('Morphing Factor')
    ax6_combined.set_ylabel('Log Loss Y|Y,X')
    ax6_combined.set_title('Model Training Log Loss Y|Y,X')
    ax6_combined.legend()
    ax6_combined.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_complete_overview.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Save detailed results
    import pandas as pd
    
    # Flatten individual runs for CSV export
    detailed_data = []
    for i, factor in enumerate(results['morph_factors']):
        for run_idx in range(len(results['te_all_runs'][i])):
            detailed_data.append({
                'morph_factor': factor,
                'run_number': run_idx + 1,
                'te_value': results['te_all_runs'][i][run_idx],
                'h_yy_value': results['h_yy_all_runs'][i][run_idx],
                'h_yyx_value': results['h_yyx_all_runs'][i][run_idx],
                'estimation_runtime': results['estimation_runtimes'][i][run_idx],
                'log_loss_yy': results['log_loss_yy_all_runs'][i][run_idx],
                'log_loss_yyx': results['log_loss_yyx_all_runs'][i][run_idx]
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(f'results/{save_prefix}_individual_runs.csv', index=False)
    
    # Summary statistics
    summary_df = pd.DataFrame({
        'morph_factor': results['morph_factors'],
        'te_mean': te_means,
        'te_std': te_stds,
        'te_reference': te_refs,
        'h_yy_mean': h_yy_means,
        'h_yy_std': h_yy_stds,
        'h_yy_reference': h_yy_refs,
        'h_yyx_mean': h_yyx_means,
        'h_yyx_std': h_yyx_stds,
        'h_yyx_reference': h_yyx_refs,
        'step_runtime': results['step_runtimes'],
        'avg_estimation_runtime': [np.mean(est_times) for est_times in results['estimation_runtimes']],
        'total_estimation_runtime': [np.sum(est_times) for est_times in results['estimation_runtimes']],
        'log_loss_yy_mean': log_loss_yy_means,
        'log_loss_yy_std': log_loss_yy_stds,
        'log_loss_yyx_mean': log_loss_yyx_means,
        'log_loss_yyx_std': log_loss_yyx_stds
    })
    
    summary_df.to_csv(f'results/{save_prefix}_summary_results.csv', index=False)
    
    # Save detailed runtime data
    runtime_data = []
    for i, factor in enumerate(results['morph_factors']):
        for run_idx in range(len(results['estimation_runtimes'][i])):
            runtime_data.append({
                'morph_factor': factor,
                'run_number': run_idx + 1,
                'estimation_runtime': results['estimation_runtimes'][i][run_idx]
            })
        # Add step total
        runtime_data.append({
            'morph_factor': factor,
            'run_number': 'step_total',
            'estimation_runtime': results['step_runtimes'][i]
        })
    
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df.to_csv(f'results/{save_prefix}_runtimes.csv', index=False)
    
    print(f"Results saved:")
    print(f"  - Individual runs: results/{save_prefix}_individual_runs.csv")
    print(f"  - Summary statistics: results/{save_prefix}_summary_results.csv")
    print(f"  - Runtime data: results/{save_prefix}_runtimes.csv")
    print(f"  - Plots saved as separate PNG files in results/ directory")


if __name__ == "__main__":
    # Run the morphing test
    print("Starting Transfer Entropy Morphing Test")
    print("="*50)
    
    # Test parameters
    morph_factors = [0, 0.25, 0.5, 0.75, 1.0]  # Number of morphing steps (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    SIMULATION_TIME = 5 * 60  # in seconds (reduce for faster testing)
    LAMBDA_X = 30  # events/sec for Poisson process
    N_ESTIMATIONS = 3  # Number of TE estimations per step
    BASE_SEED = 68
    
    # Create results directory
    os.makedirs("results", exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Configure models
    configs = {
        "model_config_yy": {
            "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 2,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 8,        # 16 Number of components for a mixture model
            "hidden_sizes": [4, 32],     # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "gru", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "ReLU",
        },
        "model_config_yyx": {
            "model_name": "LogNormMix",  # Name of the model to use, ["LogNormMix", "ExponentialMix","GompertzMix"]
            "context_size": 4,  # From 2^0 to 2^7, i.e., 1 to 128, Size of the RNN hidden vector
            "num_mix_components": 4,  # 16 Number of components for a mixture model
            "hidden_sizes": [8, 16],       # 16 Hidden sizes of the MLP for the inter-event time distribution
            "context_extractor": "lstm", # Type of RNN to use for context extraction, ["gru", "lstm", "mlp"]
            "activation_func": "GELU",
        },
        "train_config_yy": {
            "L2_weight": 1.563e-05,          # L2 regularization parameter
            "L_entropy_weight": 2.236e-07,      # Weight for the entropy regularization term
            "L_sep_weight": 1.775e-08,               # Weight for the separation regularization term
            "L_scale_weight": 1.405e-08,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "train_config_yyx": {
            "L2_weight": 7.168e-05,          # L2 regularization parameter
            "L_entropy_weight": 9.787e-07,      # Weight for the entropy regularization term
            "L_sep_weight": 1.023e-06,               # Weight for the separation regularization term
            "L_scale_weight": 1.105e-09,             # Weight for the scale regularization term
            "learning_rate": 5e-4,           # Learning rate for Adam optimizer
            "max_epochs": 1000,              # For how many epochs to train
            "display_step": 5,               # Display training statistics after every display_step
            "patience": 40,                  # After how many consecutive epochs without improvement of val loss to stop training
        },
        "data_prep_config":{
            "batch_size": 128,          # Number of sequences in a batch
            "shuffle": False,                 # Whether to shuffle the time series before splitting into train/val/test
            "total_time": SIMULATION_TIME,              # in second, Total time of the sequences
            "verbose": False
        },
        "device": device,
        "verbose": False,  # Whether to print the training statistics
        "plot_histograms": False,  # Whether to plot the conditional histograms
        "plot_pp": False,            # Whether to plot the probability - probability plots
        "history_length": 8,             # in number of bins, Length of the history to use for the model
    }

    # Run the morphing test
    start_time = time.time()
    
    results = run_morphing_test(
        morph_factors, configs,
        simulation_time=SIMULATION_TIME,
        lambda_x=LAMBDA_X,
        n_estimations=N_ESTIMATIONS,
        base_seed=BASE_SEED
    )
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"Morphing Test Completed in {total_duration/60:.2f} minutes")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 40)
    for i, factor in enumerate(results['morph_factors']):
        te_mean = np.mean(results['te_all_runs'][i])
        te_std = np.std(results['te_all_runs'][i])
        print(f"Step {i+1}: factor={factor:.2f}, TE={te_mean:.4f}±{te_std:.4f} nats/sec")
    
    # Save intensity tables as separate CSV
    intensity_data = []
    for i, (factor, table) in enumerate(zip(results['morph_factors'], results['intensity_tables'])):
        for condition, (mu, sigma) in table.items():
            condition_name = {
                (False, False): "LL", (False, True): "LS",
                (True, False): "SL", (True, True): "SS"
            }[condition]
            intensity_data.append({
                'morph_factor': factor,
                'condition': condition_name,
                'mu': mu,
                'sigma': sigma
            })
    
    intensity_df = pd.DataFrame(intensity_data)
    intensity_df.to_csv('results/morphing_test_intensity_tables.csv', index=False)
    
    # Save simulation statistics
    sim_stats_data = []
    for i, (factor, stats) in enumerate(zip(results['morph_factors'], results['simulation_stats'])):
        sim_stats_data.append({
            'morph_factor': factor,
            'n_x_events': stats['n_x_events'],
            'n_y_events': stats['n_y_events'],
            'rate_x': stats['rate_x'],
            'rate_y': stats['rate_y']
        })
    
    sim_stats_df = pd.DataFrame(sim_stats_data)
    sim_stats_df.to_csv('results/morphing_test_simulation_stats.csv', index=False)
    
    # Plot results
    plot_morphing_results(results, save_prefix="morphing_test")

    print(f"\nComplete results saved as CSV files:")
    print(f"  - Individual runs: results/morphing_test_individual_runs.csv")
    print(f"  - Summary statistics: results/morphing_test_summary_results.csv") 
    print(f"  - Runtime data: results/morphing_test_runtimes.csv")
    print(f"  - Intensity tables: results/morphing_test_intensity_tables.csv")
    print(f"  - Simulation stats: results/morphing_test_simulation_stats.csv")
    print("Analysis complete!")

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.polynomial.laguerre import laggauss
import time
from scipy.stats import lognorm
from numpy.polynomial.hermite import hermgauss
import math
from contextlib import redirect_stdout
import pandas as pd
from piecewise_lognormal import compute_reference, simulate_processes
from typing import List, Tuple, Dict
from NPEET import mi
from KSG_test import prepare_data


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

    axes[-1].set_xlabel('Inter-event Time (seconds)', fontsize=14)
    fig.text(0.02, 0.5, 'Counts', va='center', rotation='vertical', fontsize=14)

    os.makedirs(f"./results/hists/", exist_ok=True)
    hist_dir = "./results/hists/"
    if os.path.exists(hist_dir):
        for filename in os.listdir(hist_dir):
            file_path = os.path.join(hist_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    plt.savefig(f"./results/hists/cond_hist_simu_{seed}.png", dpi=300)


def create_morphed_intensity_table(morph_factor):
    """
    Create an intensity table with morphed parameters.
    """
    original_table = {
        (False, False): (-5, 0.5), (False, True): (-7, 2),
        (True, False): (-3, 0.5), (True, True): (-4, 1.5)
    }
    target_table = {
        (False, False): (-7, 2), (False, True): (-7, 2),
        (True, False): (-4, 1.5), (True, True): (-4, 1.5)
    }
    morphed_table = {}
    for key in original_table:
        orig_mu, orig_sigma = original_table[key]
        target_mu, target_sigma = target_table[key]
        morphed_mu = orig_mu + morph_factor * (target_mu - orig_mu)
        morphed_sigma = orig_sigma + morph_factor * (target_sigma - orig_sigma)
        morphed_table[key] = (morphed_mu, morphed_sigma)
    return morphed_table


def save_incremental_results(results, step_idx, save_prefix="morphing_incremental"):
    # This function remains unchanged but will now save 0s for h_yy, h_yyx, and log_loss
    # ... (function content is identical to the original)
    import pandas as pd
    
    incremental_dir = "results/incremental"
    os.makedirs(incremental_dir, exist_ok=True)
    
    current_factor = results['morph_factors'][step_idx]
    current_te_runs = results['te_all_runs'][step_idx]
    current_intensity_table = results['intensity_tables'][step_idx]
    current_sim_stats = results['simulation_stats'][step_idx]
    current_step_runtime = results['step_runtimes'][step_idx]
    current_estimation_runtimes = results['estimation_runtimes'][step_idx]
    
    individual_runs_file = f'{incremental_dir}/{save_prefix}_individual_runs.csv'
    summary_file = f'{incremental_dir}/{save_prefix}_summary.csv'
    
    detailed_data = []
    for run_idx in range(len(current_te_runs)):
        detailed_data.append({
            'morph_factor': current_factor, 'run_number': run_idx + 1,
            'te_value': current_te_runs[run_idx], 'te_ref': results['te_ref'][step_idx],
            'h_yy_ref': results['h_yy_ref'][step_idx], 'h_yyx_ref': results['h_yyx_ref'][step_idx],
            'estimation_runtime': current_estimation_runtimes[run_idx], 'intensity_table': current_intensity_table, 'simulation_stats': current_sim_stats
        })
    detailed_df = pd.DataFrame(detailed_data)
    if step_idx == 0:
        detailed_df.to_csv(individual_runs_file, index=False)
    else:
        detailed_df.to_csv(individual_runs_file, mode='a', header=False, index=False)

    summary_data = [{'morph_factor': current_factor, 'te_mean': np.mean(current_te_runs), 'te_ref': results['te_ref'][step_idx],'te_std': np.std(current_te_runs),
                     'step_runtime': current_step_runtime, 'avg_estimation_runtime': np.mean(current_estimation_runtimes),
                     'total_estimation_runtime': np.sum(current_estimation_runtimes),}]
    summary_df = pd.DataFrame(summary_data)
    if step_idx == 0:
        summary_df.to_csv(summary_file, index=False)
    else:
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)
        
    print(f"  → Step {step_idx + 1} results appended to incremental files in {incremental_dir}/")


def run_morphing_test(
    morph_factors: list, 
    configs: dict,
    simulation_time: float = 15*60,
    lambda_x: float = 30,
    n_estimations: int = 3,
    base_seed: int = 64
):
    print(f"=== Starting Morphing Test using KSG Estimator ===")
    
    morph_factors = sorted(set(morph_factors))
    assert all(0 <= mf <= 1 for mf in morph_factors), "Morph factors must be between 0 and 1"
    n_morph_steps = len(morph_factors)
    
    results = {
        'morph_factors': [], 'te_all_runs': [], 'te_ref': [], 'h_yy_ref': [], 'h_yyx_ref': [],
        'intensity_tables': [], 'simulation_stats': [], 'step_runtimes': [], 'estimation_runtimes': [],
    }

    for i, morph_factor in tqdm(enumerate(morph_factors)):
        step_start_time = time.time()
        
        print(f"\n{'='*60}\nMorphing Step {i+1}/{n_morph_steps}: factor = {morph_factor:.2f}\n{'='*60}")
        
        intensity_table = create_morphed_intensity_table(morph_factor)
        
        step_seed = base_seed + i * 100
        np.random.seed(step_seed)
        
        print(f"Simulating data with seed {step_seed}...")
        x_events, y_events, count_table, _ = simulate_processes(
            simulation_time, lambda_x, intensity_table, step_seed
        )

        h_yy_ref, h_yyx_ref, te_ref = compute_reference(count_table, intensity_table, simulation_time)
        
        sim_stats = {
            'n_x_events': len(x_events), 'n_y_events': len(y_events),
            'rate_x': len(x_events) / simulation_time, 'rate_y': len(y_events) / simulation_time,
            'count_table': count_table
        }
        
        print(f"Generated {len(y_events)} Y events and {len(x_events)} X events")
        
        print(f"Running {n_estimations} TE estimations using KSG...")
        
        te_values, h_yy_values, h_yyx_values = [], [], []
        estimation_times, log_loss_yy_values, log_loss_yyx_values = [], [], []
        
        for est_run in range(n_estimations):
            est_start_time = time.time()
            print(f"  Estimation {est_run+1}/{n_estimations}")
            
            try:
                # 1. Prepare data into historical arrays
                history_length = configs['data_prep_config']['history_length']
                x_history, y_present, y_history = prepare_data(
                    [np.array(y_events), np.array(x_events)],
                    history_length=history_length,
                    total_time=simulation_time,
                    verbose=configs['data_prep_config']['verbose']
                )
                
                # 2. Compute conditional mutual information using NPEET (KSG)
                # This returns TE in nats per event
                te_npeet_event = mi(x_history, y_present, y_history, k=3)
                
                # 3. Convert to per-second rate
                te_sec = te_npeet_event * len(y_present) / simulation_time
                
                # 4. Set placeholder values for metrics not provided by KSG
                h_yy_sec = 0.0
                h_yyx_sec = 0.0
                log_loss_yy = 0.0
                log_loss_yyx = 0.0
            
            except ValueError as e:
                print(f"    ERROR during data preparation: {e}. Skipping run.")
                te_sec, h_yy_sec, h_yyx_sec = np.nan, np.nan, np.nan
                log_loss_yy, log_loss_yyx = np.nan, np.nan

            est_end_time = time.time()
            est_runtime = est_end_time - est_start_time
            estimation_times.append(est_runtime)
            
            te_values.append(te_sec)
            h_yy_values.append(h_yy_sec)
            h_yyx_values.append(h_yyx_sec)
            log_loss_yy_values.append(log_loss_yy)
            log_loss_yyx_values.append(log_loss_yyx)
            # --- END MODIFICATION ---

            print(f"    TE_sec (KSG): {te_sec:.4f} (runtime: {est_runtime:.2f} s)")
        
        te_mean = np.nanmean(te_values)
        te_std = np.nanstd(te_values)
        step_runtime = time.time() - step_start_time
        
        print(f"Results: TE = {te_mean:.4f} ± {te_std:.4f} nats/sec")
        print(f"Step runtime: {step_runtime:.2f}s")
        
        results['morph_factors'].append(morph_factor)
        results['te_all_runs'].append(te_values.copy())
        results['intensity_tables'].append(intensity_table)
        results['simulation_stats'].append(sim_stats)
        results['step_runtimes'].append(step_runtime)
        results['estimation_runtimes'].append(estimation_times.copy())
        results['te_ref'].append(te_ref)
        results['h_yy_ref'].append(h_yy_ref)
        results['h_yyx_ref'].append(h_yyx_ref)

        save_incremental_results(results, i, save_prefix="morphing_test_ksg")
    
    return results


def plot_morphing_results_ksg(results: dict, save_prefix="morphing_ksg"):
    """
    Plots the morphing test results generated by the KSG estimator.

    This function is tailored to the 'results' dictionary from the KSG-based
    morphing test, which includes TE estimates and reference values but not
    estimated conditional entropies or log loss.
    """
    print("📊 Generating plots for KSG morphing test results...")

    # --- 1. Data Extraction ---
    morph_factors = np.array(results['morph_factors'])
    te_all_runs = results['te_all_runs']

    # Calculate means and standard deviations for estimated TE
    te_means = np.array([np.nanmean(runs) for runs in te_all_runs])
    te_stds = np.array([np.nanstd(runs) for runs in te_all_runs])

    # Extract pre-calculated reference values directly from results
    te_refs = np.array(results['te_ref'])
    h_yy_refs = np.array(results['h_yy_ref'])
    h_yyx_refs = np.array(results['h_yyx_ref'])

    # --- 2. Generate Plots ---

    # Plot 1: Transfer Entropy vs. Morphing Factor (Main Result)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(morph_factors, te_means, yerr=te_stds,
                 marker='o', linewidth=2, markersize=8, capsize=5,
                 color='blue', label='Estimated TE (KSG)')
    ax1.plot(morph_factors, te_refs, 'r--', linewidth=2, markersize=6,
             marker='s', label='Reference TE', alpha=0.8)

    # Plot individual estimation runs as scatter points
    for factor, te_runs in zip(morph_factors, te_all_runs):
        ax1.scatter([factor] * len(te_runs), te_runs, alpha=0.3, s=20, color='lightblue')

    ax1.set_xlabel('Morphing Factor', fontsize=12)
    ax1.set_ylabel('Transfer Entropy (nats/sec)', fontsize=12)
    ax1.set_title('Transfer Entropy vs. Morphing Factor (KSG Estimator)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_te_evolution.png', dpi=300, bbox_inches='tight')

    # Plot 2: Reference Conditional Entropies
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(morph_factors, h_yy_refs, 'g--', linewidth=2, marker='s',
             markersize=6, label='H(Y|Y) Reference', alpha=0.8)
    ax2.plot(morph_factors, h_yyx_refs, color='darkorange', linestyle='--',
             linewidth=2, marker='^', markersize=6, label='H(Y|Y,X) Reference', alpha=0.8)

    ax2.set_xlabel('Morphing Factor', fontsize=12)
    ax2.set_ylabel('Conditional Entropy (nats/sec)', fontsize=12)
    ax2.set_title('Reference Conditional Entropies vs. Morphing Factor', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_reference_entropies.png', dpi=300, bbox_inches='tight')

    # Plot 3: Parameter Evolution (µ parameter)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    intensity_tables = results['intensity_tables']
    ll_mu = [table[(False, False)][0] for table in intensity_tables]
    ls_mu = [table[(False, True)][0] for table in intensity_tables]
    sl_mu = [table[(True, False)][0] for table in intensity_tables]
    ss_mu = [table[(True, True)][0] for table in intensity_tables]

    ax3.plot(morph_factors, ll_mu, 'o-', label='LL (μ)', linewidth=2, markersize=6)
    ax3.plot(morph_factors, ls_mu, 's-', label='LS (μ)', linewidth=2, markersize=6)
    ax3.plot(morph_factors, sl_mu, '^-', label='SL (μ)', linewidth=2, markersize=6)
    ax3.plot(morph_factors, ss_mu, 'd-', label='SS (μ)', linewidth=2, markersize=6)
    ax3.set_xlabel('Morphing Factor', fontsize=12)
    ax3.set_ylabel('μ parameter', fontsize=12)
    ax3.set_title('Simulation Parameter Evolution', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{save_prefix}_parameter_evolution.png', dpi=300, bbox_inches='tight')

    # Plot 4: Combined Overview (2x2 Grid)
    fig4, ((ax1_c, ax2_c), (ax3_c, ax4_c)) = plt.subplots(2, 2, figsize=(16, 12))
    fig4.suptitle('KSG Morphing Test - Overview', fontsize=16)

    # Subplot 1: TE with individual points and reference
    ax1_c.errorbar(morph_factors, te_means, yerr=te_stds, marker='o', linewidth=2,
                   markersize=6, capsize=5, color='blue', label='Estimated TE (KSG)')
    ax1_c.plot(morph_factors, te_refs, 'r--', linewidth=2, marker='s',
               markersize=5, label='Reference TE', alpha=0.8)
    for factor, te_runs in zip(morph_factors, te_all_runs):
        ax1_c.scatter([factor] * len(te_runs), te_runs, alpha=0.3, s=15, color='lightblue')
    ax1_c.set_title('Transfer Entropy vs. Morphing Factor')
    ax1_c.set_ylabel('TE (nats/sec)')
    ax1_c.grid(True, alpha=0.3)
    ax1_c.legend(fontsize=10)

    # Subplot 2: Reference Conditional Entropies
    ax2_c.plot(morph_factors, h_yy_refs, 'g--', linewidth=2, marker='s',
               markersize=5, label='H(Y|Y) Reference', alpha=0.8)
    ax2_c.plot(morph_factors, h_yyx_refs, color='darkorange', linestyle='--',
               linewidth=2, marker='^', markersize=5, label='H(Y|Y,X) Reference', alpha=0.8)
    ax2_c.set_title('Reference Conditional Entropies')
    ax2_c.set_ylabel('Conditional Entropy (nats/sec)')
    ax2_c.grid(True, alpha=0.3)
    ax2_c.legend(fontsize=10)

    # Subplot 3: All individual TE runs as lines
    if te_all_runs and len(te_all_runs[0]) > 0:
        num_runs = len(te_all_runs[0])
        for run_idx in range(num_runs):
            te_line = [step_runs[run_idx] for step_runs in te_all_runs]
            ax3_c.plot(morph_factors, te_line, 'o-', alpha=0.5, linewidth=1,
                       markersize=4, label=f'Run {run_idx+1}' if run_idx < 5 else None) # Avoid clutter
    ax3_c.plot(morph_factors, te_means, 'bo-', linewidth=3, markersize=8, label='Mean TE')
    ax3_c.set_title('Individual TE Estimation Runs')
    ax3_c.set_ylabel('TE (nats/sec)')
    ax3_c.grid(True, alpha=0.3)
    ax3_c.legend()

    # Subplot 4: Variance of TE Estimation
    te_vars = [np.nanvar(runs) for runs in te_all_runs]
    ax4_c.plot(morph_factors, te_vars, 'o-', label='TE Estimation Variance',
               linewidth=2, markersize=6, color='purple')
    ax4_c.set_title('Estimation Variance vs. Morphing Factor')
    ax4_c.set_ylabel('Variance')
    ax4_c.grid(True, alpha=0.3)
    ax4_c.legend()

    for ax in [ax3_c, ax4_c]:
        ax.set_xlabel('Morphing Factor')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'results/{save_prefix}_complete_overview.png', dpi=300, bbox_inches='tight')

    # --- 3. Save Final Results to CSV ---
    summary_df = pd.DataFrame({
        'morph_factor': morph_factors,
        'te_mean': te_means,
        'te_std': te_stds,
        'te_reference': te_refs,
        'h_yy_reference': h_yy_refs,
        'h_yyx_reference': h_yyx_refs,
    })
    summary_df.to_csv(f'results/{save_prefix}_summary.csv', index=False)

    print(f"✅ Plots and summary CSV saved with prefix '{save_prefix}' in 'results/' directory.")


if __name__ == "__main__":
    print("Starting Transfer Entropy Morphing Test with KSG Estimator")
    print("="*50)
    
    # Test parameters
    morph_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    SIMULATION_TIME = 15 * 60
    LAMBDA_X = 30
    N_ESTIMATIONS = 10
    BASE_SEED = 68
    
    os.makedirs("results", exist_ok=True)

    # The configs dict is now much simpler, as we only need data_prep_config
    # The model and train configs are no longer used by the KSG estimator.
    configs = {
        "data_prep_config": {
            "history_length": 8,
            "total_time": SIMULATION_TIME,
            "verbose": False
        },
        "verbose": False,
    }

    start_time = time.time()
    results = run_morphing_test(
        morph_factors, configs,
        simulation_time=SIMULATION_TIME,
        lambda_x=LAMBDA_X,
        n_estimations=N_ESTIMATIONS,
        base_seed=BASE_SEED
    )
    total_duration = (time.time() - start_time) / 60
    
    print(f"\n{'='*60}\nMorphing Test Completed in {total_duration:.2f} minutes\n{'='*60}")
    
    plot_morphing_results_ksg(results, save_prefix="morphing_ksg")

    print("Analysis complete! Results saved in the 'results/' directory.")
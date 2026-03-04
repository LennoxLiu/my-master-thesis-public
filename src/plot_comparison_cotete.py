import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(file_cotete, file_diff):
    # Load data
    df_cotete = pd.read_csv(file_cotete)
    df_diff = pd.read_csv(file_diff)

    # --- Process Cotete Data ---
    # Group by history_length and calculate mean and std
    cotete_grouped = df_cotete.groupby('history_length').agg(['mean', 'std'])
    cotete_x = cotete_grouped.index
    cotete_te_mean = cotete_grouped['transfer_entropy']['mean']
    cotete_te_std = cotete_grouped['transfer_entropy']['std']
    cotete_rt_mean = cotete_grouped['runtime_seconds']['mean']
    cotete_rt_std = cotete_grouped['runtime_seconds']['std']

    # --- Process Diff History Data ---
    # Group by history(intervals) and calculate mean and std
    diff_grouped = df_diff.groupby('history(intervals)').agg(['mean', 'std'])
    diff_x = diff_grouped.index
    diff_te_mean = diff_grouped['TE(nats/sec)']['mean']
    diff_te_std = diff_grouped['TE(nats/sec)']['std']
    diff_rt_mean = diff_grouped['runtime(sec)']['mean']
    diff_rt_std = diff_grouped['runtime(sec)']['std']

    # Combine all unique x-values from both datasets to ensure every point gets a tick
    all_ticks = [0]+sorted(list(set(cotete_x).union(set(diff_x))))
    
    # --- Plot 1: Transfer Entropy ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(cotete_x, cotete_te_mean, yerr=cotete_te_std, fmt='-o', 
                 label='CoTETE', capsize=5, alpha=0.8)
    plt.errorbar(diff_x, diff_te_mean, yerr=diff_te_std, fmt='-s', 
                 label='Our estimator', capsize=5, alpha=0.8)
    # Reference Line (Line in Firebrick Red)
    # Changed color to a distinct, professional red
    plt.axhline(y=0.5076, color='#cc0000', linestyle='--', linewidth=1.2, label='Reference TE value')

    
    plt.xlabel('History Length')
    plt.xticks(all_ticks)
    plt.xscale('log', base=2)
    plt.ylabel('Transfer Entropy (nats/sec)')
    plt.ylim(bottom=0)
    # plt.title('Transfer Entropy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/comparison_te_plot.png', dpi=300)
    plt.show()

    # --- Plot 2: Runtime ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(cotete_x, cotete_rt_mean, yerr=cotete_rt_std, fmt='-o', 
                 label='CoTETE', capsize=5, alpha=0.8)
    plt.errorbar(diff_x, diff_rt_mean, yerr=diff_rt_std, fmt='-s', 
                 label='Our estimator', capsize=5, alpha=0.8)
    plt.xlabel('History Length')
    plt.xticks(all_ticks)
    # plt.xscale('log', base=2)
    plt.ylabel('Runtime (seconds)')
    # plt.title('Runtime Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/comparison_runtime_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_comparison('results/cotete-seed=52/cotete_results-2powers.csv', 'results/cotete-seed=52/diff_history_results.csv')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    file_path = "results/te_results_hpc.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Create descriptive labels
    df = df.copy()
    df['source_label'] = df['source_group_id'] + " " + df['source_neuron_id'].astype(str)
    df['target_label'] = df['target_group_id'] + " " + df['target_neuron_id'].astype(str)

    # Filter directions
    bc_to_pom = df[(df['source_group_id'] == 'BC') & (df['target_group_id'] == 'POm')]
    pom_to_bc = df[(df['source_group_id'] == 'POm') & (df['target_group_id'] == 'BC')]

    # Pivot data
    pivot_bc_pom = bc_to_pom.pivot(index='target_label', columns='source_label', values='te mean (nats per second)')
    pivot_pom_bc = pom_to_bc.pivot(index='target_label', columns='source_label', values='te mean (nats per second)')

    # Sort labels for consistent ordering
    pivot_bc_pom = pivot_bc_pom.sort_index(axis=0).sort_index(axis=1)
    pivot_pom_bc = pivot_pom_bc.sort_index(axis=0).sort_index(axis=1)

    # Determine global scale for shared colorbar
    vmin = df['te mean (nats per second)'].min()
    vmax = df['te mean (nats per second)'].max()

    # Create figure with two subplots and a space for the colorbar
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Plot BC -> POm
    sns.heatmap(
        pivot_bc_pom, 
        ax=axes[0],
        annot=True, 
        fmt=".1f", 
        cmap="YlGnBu",
        vmin=vmin, 
        vmax=vmax,
        cbar=False
    )
    axes[0].set_title("BC to POm")
    axes[0].set_xlabel("Source (BC)")
    axes[0].set_ylabel("Target (POm)")

    # Plot POm -> BC
    sns.heatmap(
        pivot_pom_bc, 
        ax=axes[1],
        annot=True, 
        fmt=".1f", 
        cmap="YlGnBu",
        vmin=vmin, 
        vmax=vmax,
        cbar=False
    )
    axes[1].set_title("POm to BC")
    axes[1].set_xlabel("Source (POm)")
    axes[1].set_ylabel("Target (BC)")

    # Add a single shared colorbar
    mappable = axes[0].collections[0]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(mappable, cax=cbar_ax, label='TE (nats/sec)')

    plt.suptitle("Comparative Transfer Entropy Heatmaps", fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.3)
    
    output_filename = "results/te_heatmaps.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved: {output_filename}")

if __name__ == "__main__":
    main()
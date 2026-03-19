import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(file_path):
    """Loads and prepares the dataframe with descriptive labels."""
    df = pd.read_csv(file_path)
    df = df.copy()
    df['source_label'] = df['source_group_id'] + " " + df['source_neuron_id'].astype(str)
    df['target_label'] = df['target_group_id'] + " " + df['target_neuron_id'].astype(str)
    return df

def draw_combined_te_heatmap(df, output_filename, title):
    """
    Generates a figure with two side-by-side heatmaps (BC->POm and POm->BC)
    sharing a single color bar.
    """
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

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot BC -> POm
    sns.heatmap(
        pivot_bc_pom, ax=axes[0], annot=True, fmt=".1f", 
        cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=False
    )
    axes[0].set_title("BC to POm")
    axes[0].set_xlabel("Source (BC)")
    axes[0].set_ylabel("Target (POm)")

    # Plot POm -> BC
    sns.heatmap(
        pivot_pom_bc, ax=axes[1], annot=True, fmt=".1f", 
        cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=False
    )
    axes[1].set_title("POm to BC")
    axes[1].set_xlabel("Source (POm)")
    axes[1].set_ylabel("Target (BC)")

    # Add shared colorbar
    mappable = axes[0].collections[0]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(mappable, cax=cbar_ax, label='TE (nats/sec)')

    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(right=0.9, wspace=0.3)
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")

def draw_combined_te_heatmap_zeroed(df, output_filename, title):
    """
    Filters out negative TE values by clipping them to zero, 
    then calls the standard plotting function.
    """
    df_clipped = df.copy()
    # Replace negative values with 0
    df_clipped['te mean (nats per second)'] = df_clipped['te mean (nats per second)'].clip(lower=0)
    
    # Update title to reflect clipping
    full_title = f"{title} (Negative values set to 0)"
    
    draw_combined_te_heatmap(df_clipped, output_filename, full_title)

if __name__ == "__main__":
    DATA_PATH = "results/te_results_hpc.csv"
    
    try:
        # 1. Prepare Data
        data = prepare_data(DATA_PATH)
        
        # 2. Draw standard heatmap
        draw_combined_te_heatmap(
            data, 
            "results/te_heatmap_standard.png", 
            "Transfer Entropy Comparison"
        )
        
        # 3. Draw zero-filtered heatmap
        draw_combined_te_heatmap_zeroed(
            data, 
            "results/te_heatmap_zeroed.png", 
            "Transfer Entropy Comparison"
        )
        
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
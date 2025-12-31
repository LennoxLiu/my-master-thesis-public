import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
# We'll still use the same data loading function from before
def load_and_prepare_data(csv_filepath='quadrature_results.csv'):
    """
    Loads quadrature data and prepares it for plotting by calculating
    differences and melting the DataFrame into a long format.
    """
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at '{os.path.abspath(csv_filepath)}'")
        return None
    try:
        data = pd.read_csv(csv_filepath, dtype=np.float32)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return None

    metrics = ["TE", "H_yy", "H_yyx"]
    methods = ["GHQ", "SinhSinh", "Sinh"]
    
    all_diffs = []
    for metric in metrics:
        analytical_col = f"{metric}_Analytical"
        if analytical_col not in data.columns:
            continue
        for method in methods:
            method_col = f"{metric}_{method}"
            if method_col in data.columns:
                diff = np.abs(data[method_col] - data[analytical_col])
                temp_df = pd.DataFrame({
                    'Error': diff, 'Method': method, 'Metric': metric
                })
                all_diffs.append(temp_df)

    if not all_diffs:
        return None
    return pd.concat(all_diffs).dropna()

def create_error_plot(data, methods_to_plot, title, filename):
    """
    Creates and saves a faceted boxplot with individual data points (stripplot).

    Args:
        data (pd.DataFrame): Tidy DataFrame containing the error data.
        methods_to_plot (list): A list of method names to include in the plot.
        title (str): The main title for the plot.
        filename (str): The filename to save the plot.
    """
    # Filter the data to include only the desired methods
    plot_data = data[data['Method'].isin(methods_to_plot)]

    if plot_data.empty:
        print(f"Warning: No data found for methods {methods_to_plot}. Skipping plot '{filename}'.")
        return

    # --- 2. Create the Plot using Seaborn's catplot ---
    # `catplot` is a high-level interface for drawing categorical plots onto a FacetGrid.
    g = sns.catplot(
        data=plot_data,
        x='Method',
        y='Error',
        col='Metric',         # Create subplots for each metric
        kind='box',           # Use a boxplot as the main plot type
        height=6,
        aspect=1,
        sharey=False,         # Give each subplot its own y-axis scale
        palette='colorblind',
        legend=False          # We will create a custom title instead
    )

    # Overlay a stripplot to show individual data points with jitter
    g.map_dataframe(
        sns.stripplot,
        x='Method',
        y='Error',
        color='black',
        alpha=0.3,
        size=3
    )
    
    # --- 3. Customize and Finalize the Plot ---
    # Add a horizontal line at y=0 for reference
    g.map(plt.axhline, y=0, ls='--', c='red', lw=1.2)

    # Set titles and labels
    g.fig.suptitle(title, y=1.03, fontsize=18, fontweight='bold')
    g.set_axis_labels("", "Difference from Analytical Value")
    g.set_titles("Metric: {col_name}", size=14)
    g.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- 4. Save and Show ---
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")
        plt.show()
    except Exception as e:
        print(f"Error saving or showing plot: {e}")


if __name__ == '__main__':
    # Prepare the data once
    tidy_data = load_and_prepare_data('./results/quadrature_results.csv')

    if tidy_data is not None:
        # Plot 1: All methods
        create_error_plot(
            data=tidy_data,
            methods_to_plot=["GHQ", "SinhSinh", "Sinh"],
            title='Comparison of All Quadrature Method Errors',
            filename='./results/quadrature_errors_all_methods.png'
        )

        # Plot 2: Subset of methods, demonstrating reusability
        create_error_plot(
            data=tidy_data,
            methods_to_plot=["SinhSinh", "Sinh"],
            title='Focused Comparison of Sinh-based Method Errors',
            filename='./results/quadrature_errors_sinh_methods.png'
        )
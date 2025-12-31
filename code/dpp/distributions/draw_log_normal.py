import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def plot_lognormal_pdf(mu_sigma_pairs, x_range=(0, 5), num_points=1000):
    """
    Plots the PDF of log-normal distributions for multiple (mu, sigma) pairs.

    Parameters:
        mu_sigma_pairs (list of tuples): Each tuple contains (mu, sigma).
        x_range (tuple): The range of x values to compute the PDF over.
        num_points (int): Number of points to use in the x-axis.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    
    plt.figure(figsize=(10, 6))

    for mu, sigma in mu_sigma_pairs:
        s = sigma  # shape parameter in scipy's lognorm
        scale = np.exp(mu)
        pdf_values = lognorm.pdf(x, s=s, scale=scale)
        label = f'$\\mu$={mu}, $\\sigma$={sigma:.2f}'
        plt.plot(x, pdf_values, label=label)

    plt.title('Log-Normal Distribution PDFs')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
mu_comb = np.arange(-8, 2, 0.25)
sigma_comb = np.exp(np.arange(-2, 0, 0.5))
parameter_combinations = [(mu, sigma) for mu in mu_comb for sigma in sigma_comb]

plot_lognormal_pdf(parameter_combinations)

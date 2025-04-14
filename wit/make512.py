import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib as mpl
from matplotlib.patches import Rectangle

# Set high-quality plot parameters without LaTeX
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 12
})

# Set figure DPI for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# Define the 4-parameter logistic function
def four_param_logistic(x, a, b, c, d):
    """
    4-parameter logistic function.

    Parameters:
    - x: Input (dimension)
    - a: Lower asymptote (minimum value)
    - b: Hill slope
    - c: EC50 (inflection point)
    - d: Upper asymptote (maximum value)

    Returns: y value according to the 4PL model
    """
    return d + (a - d) / (1 + (x/c)**b)

# Load the data
data = pd.read_csv('precision_copy.csv')

# Define metrics to plot
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
titles = ['Precision@5', 'Precision@10', 'Precision@15']

# Store EC50 values, confidence intervals, and goodness-of-fit metrics
ec50_results = []

# Fit models for each metric first to get EC50 values and other parameters
for i, (label, title) in enumerate(zip(labels, titles)):
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Get min/max values for bounds
    min_y = min(y_data)
    max_y = max(y_data)
    
    # Set x-axis limits with small padding
    x_min = min(x_data) - 2
    x_max = max(x_data) + 2

    try:
        # Set bounds to constrain the asymptotes to reasonable values
        bounds = (
            [min_y * 0.95, 0.1, 10, max_y * 0.95],  # Lower bounds
            [min_y * 1.05, 5, 100, max_y * 1.05]    # Upper bounds
        )
        
        # Perform the curve fit with bounds
        popt, pcov = curve_fit(four_param_logistic, x_data, y_data, bounds=bounds)

        # Get parameter values
        a, b, c, d = popt

        # Calculate standard errors and 95% confidence intervals
        perr = np.sqrt(np.diag(pcov))
        ci_95 = perr * 1.96  # 95% confidence interval

        # Calculate fitted values for the original x data for R² and RMSE
        y_fit_original = four_param_logistic(x_data, *popt)

        # Calculate R-squared
        r_squared = r2_score(y_data, y_fit_original)

        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(mean_squared_error(y_data, y_fit_original))

        # Store EC50 results for reporting
        ec50_results.append({
            'metric': label,
            'EC50': c,
            'EC50_CI_lower': c - ci_95[2],
            'EC50_CI_upper': c + ci_95[2],
            'params': {
                'a (lower asymptote)': (a, a - ci_95[0], a + ci_95[0]),
                'b (Hill slope)': (b, b - ci_95[1], b + ci_95[1]),
                'c (EC50)': (c, c - ci_95[2], c + ci_95[2]),
                'd (upper asymptote)': (d, d - ci_95[3], d + ci_95[3])
            },
            'goodness_of_fit': {
                'R_squared': r_squared,
                'RMSE': rmse
            },
            'popt': popt  # Store the parameters for later use
        })

    except Exception as e:
        print(f"Error fitting {label}: {e}")
        # Add empty placeholder to maintain indexing
        ec50_results.append(None)

# Load the data from precision.csv
data = pd.read_csv('precision_copy.csv')

# Filter dimensions between 300 and 512
# data = data[(data['dimension'] >= 320) & (data['dimension'] <= 512)]

# Define metrics to plot
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
titles = ['Precision@5', 'Precision@10', 'Precision@15']

# Group 5: Data points AND blue fitting curve with y axis starting from 0
for i, (label, title) in enumerate(zip(labels, titles)):
    if ec50_results[i] is None:
        print(f"Skipping {label} due to fitting error")
        continue
        
    # Create a new figure
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Get min/max values for bounds
    x_min = min(x_data) - 2
    x_max = max(x_data) + 2
    
    # Create x values for smooth curve plotting
    x_smooth = np.linspace(x_min, x_max, 1000)
    
    # Plot the data points
    ax.scatter(x_data, y_data, color='black', marker='o', s=10, alpha=0.8, label='Data points')
    
    # Get parameters and goodness-of-fit values
    popt = ec50_results[i]['popt']
    a, b, c, d = popt
    r_squared = ec50_results[i]['goodness_of_fit']['R_squared']
    rmse = ec50_results[i]['goodness_of_fit']['RMSE']
    
    # Create the fitted curve values
    y_fit = four_param_logistic(x_smooth, *popt)
    
    # Plot the fitted curve in blue
    ax.plot(x_smooth, y_fit, color='blue', linestyle='-', linewidth=2.0, alpha=0.8, label='Fitted curve')
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Precision')
    
    # Set title
    k_value = label.split('_')[-1]
    ax.set_title(f'Precision@{k_value}')
    
    # Set y-axis to start from 0 and end at 1.05
    ax.set_ylim(0.6, 1.05)
    
    # Set the x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Adjust layout - give more space on the right side
    plt.subplots_adjust(left=0.15, right=0.90, top=0.9, bottom=0.15)
    
    # Save the figure
    k_value = label.split('_')[-1]
    fig.savefig(f'data_and_fit({k_value}).png', dpi=600, bbox_inches='tight')
    
    plt.close(fig)

# GROUP 1: Histograms for each precision metric
for i, (label, title) in enumerate(zip(labels, titles)):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Set x-axis limits
    x_min = min(x_data) - 2
    x_max = max(x_data) + 2
    
    # Calculate the width for the bars
    width = 1.8  # Slightly narrower than the step size to have small gaps
    
    # Plot the histogram bars
    bars = ax.bar(x_data, y_data, width=width, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Precision')
    
    # Set title
    k_value = label.split('_')[-1]
    ax.set_title(f'Precision@{k_value} Histogram')
    
    # Set y-axis to start from 0 and end at 1.05
    ax.set_ylim(0, 1.05)
    
    # Set the x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Add minor ticks for y-axis only (x-axis would be too crowded)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    
    # Set major x-ticks at intervals to avoid overcrowding
    x_tick_interval = 20  # Adjust this value for readability
    x_ticks = np.arange(min(data['dimension']), max(data['dimension']) + 1, x_tick_interval)
    ax.set_xticks(x_ticks)
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(f'histogram_precision({k_value}).png', dpi=600, bbox_inches='tight')
    
    plt.close(fig)

# GROUP 1: Histograms for each precision metric
for i, (label, title) in enumerate(zip(labels, titles)):
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Set x-axis limits
    x_min = min(x_data) - 2
    x_max = max(x_data) + 2
    
    # Calculate the width for the bars
    width = 1.8  # Slightly narrower than the step size to have small gaps
    
    # Plot the histogram bars
    bars = ax.bar(x_data, y_data, width=width, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Precision')
    
    # Set title
    k_value = label.split('_')[-1]
    ax.set_title(f'Precision@{k_value} Histogram')
    
    # Set y-axis to start from 0 and end at 1.05
    ax.set_ylim(0.9, 1.02)
    
    # Set the x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Add minor ticks for y-axis only (x-axis would be too crowded)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    
    # Set major x-ticks at intervals to avoid overcrowding
    x_tick_interval = 20  # Adjust this value for readability
    x_ticks = np.arange(min(data['dimension']), max(data['dimension']) + 1, x_tick_interval)
    ax.set_xticks(x_ticks)
    
    # Add grid for y-axis only
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(f'histogram({k_value}).png', dpi=600, bbox_inches='tight')
    
    plt.close(fig)

print("All 6 plots have been generated successfully.")
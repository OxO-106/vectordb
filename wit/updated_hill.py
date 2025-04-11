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
data = pd.read_csv('data/precision.csv')

# Define metrics to plot
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
titles = ['Precision@5', 'Precision@10', 'Precision@15']

# Store EC50 values, confidence intervals, and goodness-of-fit metrics
ec50_results = []

# Create a figure for each precision metric
for i, (label, title) in enumerate(zip(labels, titles)):
    # Create a new figure with appropriate size for a paper column
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Get min/max values for bounds
    min_y = min(y_data)
    max_y = max(y_data)
    
    # Set x-axis limits with small padding
    x_min = min(x_data) - 2
    x_max = max(x_data) + 2
    
    # Create x values for smooth curve plotting (use more points for smoother curve)
    x_smooth = np.linspace(x_min, x_max, 1000)

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

        # Create the fitted curve values for the smooth x range
        y_fit = four_param_logistic(x_smooth, *popt)

        # Calculate fitted values for the original x data for R² and RMSE
        y_fit_original = four_param_logistic(x_data, *popt)

        # Calculate R-squared
        r_squared = r2_score(y_data, y_fit_original)

        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(mean_squared_error(y_data, y_fit_original))

        # Plot the data points
        ax.scatter(x_data, y_data, color='black', marker='o', s=25, alpha=0.8)
        
        # Plot the fitted curve with clear styling
        ax.plot(x_smooth, y_fit, color='blue', linestyle='-', linewidth=2.0)

        # Mark the EC50 point (where the curve reaches its half-maximal value)
        ec50_y = four_param_logistic(c, *popt)
        
        # Add vertical and horizontal lines to mark the EC50 point
        ax.axvline(x=c, color='gray', linestyle='--', alpha=0.7, linewidth=1.0)
        ax.axhline(y=ec50_y, color='gray', linestyle='--', alpha=0.7, linewidth=1.0)
        
        # Add an X marker at the EC50 intersection
        ax.plot(c, ec50_y, 'x', color='gray', markersize=10, markeredgewidth=2, alpha=0.8)
        
        # MODIFIED: Improved equation positioning in the white space
        # Use relative positioning instead of absolute to ensure proper placement in the bottom right
        
        # Create the equation text with the actual parameter values
        # eq_text = f"$f(x) = {d:.2f} + \\frac{{{a:.2f} - {d:.2f}}}{{1 + (\\frac{{x}}{{{c:.2f}}})^{{{b:.2f}}}}}$\n$R^2 = {r_squared:.4f}$"
        
        # Position the text in the bottom right corner using axes coordinates (0-1 range)
        # This places it at 75% of x-axis width and 20% of y-axis height
        # ax.text(0.75, 0.2, eq_text, fontsize=8, 
        #         transform=ax.transAxes,  # Use axes coordinates
        #         ha='right',  # Horizontal alignment: right
        #         va='bottom',  # Vertical alignment: bottom
        #         bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', 
        #                  boxstyle="round,pad=0.5"))

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
            }
        })

    except Exception as e:
        print(f"Error fitting {label}: {e}")
        continue
    
    # Add thicker horizontal asymptote at y=1 without text label
    #ax.axhline(y=1.0, color='red', linestyle='-', alpha=0.8, linewidth=2.0)
    
    # Set axis labels
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Precision')
    
    # Set title
    ax.set_title(title)
    
    # Set tight grid for professional appearance
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Set y-axis to start from 0 and end at 1.05
    ax.set_ylim(0, 1.05)
    
    # Set the x-axis limits
    ax.set_xlim(x_min, x_max)
    
    # Add minor ticks
    ax.minorticks_on()
    
    # Use a different approach than tight_layout to ensure everything fits
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    
    # Save the figure with a descriptive name and high resolution
    fig.savefig(f'{label}_4pl_sigmod.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{label}_4pl_sigmod.png', dpi=600, bbox_inches='tight')
    
    # Print detailed parameter information to the console
    # print(f"\n=== {title} ===")
    # print(f"  EC50 (inflection point): {c:.4f}")
    # print(f"  95% Confidence Interval: ({ec50_results[i]['EC50_CI_lower']:.4f}, {ec50_results[i]['EC50_CI_upper']:.4f})")
    
    # print("\n  All 4PL Parameters (value, lower CI, upper CI):")
    # for param_name, (value, ci_lower, ci_upper) in ec50_results[i]['params'].items():
    #     print(f"  {param_name}: {value:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")
    
    # print("\n  Goodness of Fit:")
    # print(f"  - R-squared: {ec50_results[i]['goodness_of_fit']['R_squared']:.6f}")
    # print(f"  - RMSE: {ec50_results[i]['goodness_of_fit']['RMSE']:.6f}")

# If all three metrics were successfully fitted, do a comparative analysis
# if len(ec50_results) == 3:
#     print("\n===== Comparative EC50 Analysis =====")
#     ec50_values = [result['EC50'] for result in ec50_results]
#     ec50_labels = [result['metric'] for result in ec50_results]
#     r2_values = [result['goodness_of_fit']['R_squared'] for result in ec50_results]
#     rmse_values = [result['goodness_of_fit']['RMSE'] for result in ec50_results]

#     min_ec50_idx = np.argmin(ec50_values)
#     max_ec50_idx = np.argmax(ec50_values)
#     best_fit_idx = np.argmax(r2_values)

#     print(f"Lowest EC50: {ec50_labels[min_ec50_idx]} ({ec50_values[min_ec50_idx]:.4f})")
#     print(f"Highest EC50: {ec50_labels[max_ec50_idx]} ({ec50_values[max_ec50_idx]:.4f})")
#     print(f"Best model fit: {ec50_labels[best_fit_idx]} (R² = {r2_values[best_fit_idx]:.4f}, RMSE = {rmse_values[best_fit_idx]:.4f})")

#     if max(ec50_values) - min(ec50_values) > 1.0:
#         print("\nThere is a substantial difference between EC50 values across metrics.")
#         print("This suggests that different precision metrics have different sensitivity to dimension.")
#     else:
#         print("\nThe EC50 values are relatively close across different metrics.")
#         print("This suggests consistent behavior across different precision metrics.")
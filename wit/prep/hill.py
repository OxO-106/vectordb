import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

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

# Prepare figure for plotting
plt.figure(figsize=(12, 8))
colors = ['blue', 'green', 'red']
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
markers = ['o', 's', '^']

# Store EC50 values, confidence intervals, and goodness-of-fit metrics
ec50_results = []

# Create x values for smooth curve plotting
x_smooth = np.linspace(min(data['dimension']) - 0.5, max(data['dimension']) + 2, 1000)

# Process each precision metric
for i, label in enumerate(labels):
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values

    # Initial parameter guess [a, b, c, d]
    # a: min value, d: max value, c: middle x, b: slope (positive for increasing)
    p0 = [min(y_data), 1, np.median(x_data), max(y_data)]

    try:
        # Perform the curve fit
        popt, pcov = curve_fit(four_param_logistic, x_data, y_data, p0=p0)

        # Get parameter values
        a, b, c, d = popt

        # Calculate standard errors and 95% confidence intervals
        perr = np.sqrt(np.diag(pcov))
        ci_95 = perr * 1.96  # 95% confidence interval

        # Create the fitted curve
        y_fit = four_param_logistic(x_smooth, *popt)

        # Calculate fitted values for the original x data for R² and RMSE
        y_fit_original = four_param_logistic(x_data, *popt)

        # Calculate R-squared
        r_squared = r2_score(y_data, y_fit_original)

        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(mean_squared_error(y_data, y_fit_original))

        # Plot the data and fit
        plt.scatter(x_data, y_data, color=colors[i], marker=markers[i], s=60, label=f'{label} (data)')
        plt.plot(x_smooth, y_fit, color=colors[i], linestyle='-', linewidth=2, label=f'{label} (4PL fit)')

        # Mark the EC50 point
        ec50_y = four_param_logistic(c, *popt)
        plt.scatter(c, ec50_y, color=colors[i], marker='x', s=100, zorder=5)
        plt.axvline(x=c, color=colors[i], linestyle='--', alpha=0.3)
        plt.axhline(y=ec50_y, color=colors[i], linestyle='--', alpha=0.3)

        # Store EC50 results
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

# Finalize the plot
plt.xlabel('Dimension', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('4-Parameter Logistic Model Fitting for Precision Data', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add annotations for EC50 values and goodness-of-fit metrics
y_pos = 0.25
for i, result in enumerate(ec50_results):
    plt.annotate(
        f"{result['metric']}:\n" +
        f"EC50 = {result['EC50']:.2f} (95% CI: {result['EC50_CI_lower']:.2f}-{result['EC50_CI_upper']:.2f})\n" +
        f"R² = {result['goodness_of_fit']['R_squared']:.4f}, RMSE = {result['goodness_of_fit']['RMSE']:.4f}",
        xy=(0.05, y_pos - i*0.07),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8)
    )

plt.tight_layout()
plt.savefig('precision_4pl_fit.png', dpi=300)
plt.show()

# Print detailed EC50 analysis and parameter information
print("\n===== EC50 Analysis =====")
for result in ec50_results:
    print(f"\n{result['metric']}:")
    print(f"  EC50 (inflection point): {result['EC50']:.4f}")
    print(f"  95% Confidence Interval: ({result['EC50_CI_lower']:.4f}, {result['EC50_CI_upper']:.4f})")

    print("\n  All 4PL Parameters (value, lower CI, upper CI):")
    for param_name, (value, ci_lower, ci_upper) in result['params'].items():
        print(f"  {param_name}: {value:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f})")

    # Goodness of fit metrics
    print("\n  Goodness of Fit:")
    print(f"  - R-squared: {result['goodness_of_fit']['R_squared']:.6f}")
    print(f"  - RMSE: {result['goodness_of_fit']['RMSE']:.6f}")

    # Interpretation
    print("\n  Interpretation:")
    print(f"  - The dimension at which {result['metric']} reaches its half-maximal value is approximately {result['EC50']:.2f}")
    print(f"  - The lower asymptote (minimum precision) is approximately {result['params']['a (lower asymptote)'][0]:.4f}")
    print(f"  - The upper asymptote (maximum precision) is approximately {result['params']['d (upper asymptote)'][0]:.4f}")
    if result['params']['b (Hill slope)'][0] > 0:
        print(f"  - The positive Hill slope ({result['params']['b (Hill slope)'][0]:.4f}) indicates that precision increases with dimension")
    else:
        print(f"  - The negative Hill slope ({result['params']['b (Hill slope)'][0]:.4f}) indicates that precision decreases with dimension")
    print(f"  - Steepness of the curve is determined by the Hill slope - higher absolute values indicate sharper transitions")

    # Interpretation of goodness of fit
    r2 = result['goodness_of_fit']['R_squared']
    if r2 > 0.95:
        print(f"  - The model fits the data extremely well (R² = {r2:.4f})")
    elif r2 > 0.9:
        print(f"  - The model fits the data very well (R² = {r2:.4f})")
    elif r2 > 0.8:
        print(f"  - The model fits the data well (R² = {r2:.4f})")
    elif r2 > 0.6:
        print(f"  - The model fits the data moderately well (R² = {r2:.4f})")
    else:
        print(f"  - The model does not fit the data well (R² = {r2:.4f})")

# Comparative analysis of EC50 across metrics
if len(ec50_results) > 1:
    print("\n===== Comparative EC50 Analysis =====")
    ec50_values = [result['EC50'] for result in ec50_results]
    ec50_labels = [result['metric'] for result in ec50_results]
    r2_values = [result['goodness_of_fit']['R_squared'] for result in ec50_results]
    rmse_values = [result['goodness_of_fit']['RMSE'] for result in ec50_results]

    min_ec50_idx = np.argmin(ec50_values)
    max_ec50_idx = np.argmax(ec50_values)
    best_fit_idx = np.argmax(r2_values)

    print(f"Lowest EC50: {ec50_labels[min_ec50_idx]} ({ec50_values[min_ec50_idx]:.4f})")
    print(f"Highest EC50: {ec50_labels[max_ec50_idx]} ({ec50_values[max_ec50_idx]:.4f})")
    print(f"Best model fit: {ec50_labels[best_fit_idx]} (R² = {r2_values[best_fit_idx]:.4f}, RMSE = {rmse_values[best_fit_idx]:.4f})")

    if max(ec50_values) - min(ec50_values) > 1.0:
        print("\nThere is a substantial difference between EC50 values across metrics.")
        print("This suggests that different precision metrics have different sensitivity to dimension.")
    else:
        print("\nThe EC50 values are relatively close across different metrics.")
        print("This suggests consistent behavior across different precision metrics.")

    print("\n===== Goodness of Fit Summary =====")
    for i, label in enumerate(ec50_labels):
        print(f"{label}: R² = {r2_values[i]:.4f}, RMSE = {rmse_values[i]:.4f}")
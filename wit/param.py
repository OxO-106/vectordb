import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

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

# Define metrics to analyze
labels = ['precision_at_5', 'precision_at_10', 'precision_at_15']
titles = ['Precision@5', 'Precision@10', 'Precision@15']

# Store parameters and fitting results
fitting_results = []

# Fit models for each metric
print("4-Parameter Logistic Function Fitting Results:")
print("=" * 50)

for i, (label, title) in enumerate(zip(labels, titles)):
    # Extract x and y data
    x_data = data['dimension'].values
    y_data = data[label].values
    
    # Get min/max values for bounds
    min_y = min(y_data)
    max_y = max(y_data)

    try:
        # Set bounds to constrain the asymptotes to reasonable values
        bounds = (
            [min_y * 0.95, 0.1, 10, max_y * 0.95],  # Lower bounds
            [min_y * 1.05, 5, 100, max_y * 1.05]     # Upper bounds
        )
        
        # Perform the curve fit with bounds
        popt, pcov = curve_fit(four_param_logistic, x_data, y_data, bounds=bounds)

        # Get parameter values
        a, b, c, d = popt

        # Calculate standard errors and 95% confidence intervals
        perr = np.sqrt(np.diag(pcov))
        ci_95 = perr * 1.96  # 95% confidence interval

        # Calculate fitted values for the original x data
        y_fit_original = four_param_logistic(x_data, *popt)

        # Calculate R-squared
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        ss_res = np.sum((y_data - y_fit_original)**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((y_data - y_fit_original)**2))

        # Print results
        print(f"\n{title} Parameters:")
        print(f"a (Lower asymptote): {a:.6f} ± {ci_95[0]:.6f}")
        print(f"b (Hill slope): {b:.6f} ± {ci_95[1]:.6f}")
        print(f"c (EC50): {c:.6f} ± {ci_95[2]:.6f}")
        print(f"d (Upper asymptote): {d:.6f} ± {ci_95[3]:.6f}")
        print(f"R²: {r_squared:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        # Store results in a more structured format
        fitting_results.append({
            'metric': label,
            'parameters': {
                'a': a,
                'b': b,
                'c': c,
                'd': d
            },
            'confidence_intervals': {
                'a_ci': ci_95[0],
                'b_ci': ci_95[1],
                'c_ci': ci_95[2],
                'd_ci': ci_95[3]
            },
            'goodness_of_fit': {
                'R_squared': r_squared,
                'RMSE': rmse
            }
        })

    except Exception as e:
        print(f"Error fitting {label}: {e}")

# Print a nicely formatted table of all parameters
print("\n\nParameter Summary Table:")
print("=" * 80)
print(f"{'Metric':<15} {'a':<12} {'b':<12} {'c':<12} {'d':<12} {'R²':<12} {'RMSE':<12}")
print("-" * 80)

for result in fitting_results:
    metric = result['metric'].split('_')[-1]  # Extract k value
    p = result['parameters']
    gof = result['goodness_of_fit']
    print(f"Precision@{metric:<6} {p['a']:<12.4f} {p['b']:<12.4f} {p['c']:<12.4f} {p['d']:<12.4f} {gof['R_squared']:<12.4f} {gof['RMSE']:<12.4f}")
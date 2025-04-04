import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from precision4.csv
data = pd.read_csv('data/precision4.csv')
x = data['dimension'].values
y10 = data['precision_at_10'].values

# Define candidate functions to fit
def logistic(x, L, k, x0, b):
    """Logistic function: L / (1 + exp(-k * (x - x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b

def exp_approach(x, a, b, c):
    """Exponential approach: a * (1 - exp(-b * x)) + c"""
    return a * (1 - np.exp(-b * x)) + c

# Define a logarithmic function
def log_func(x, a, b, c):
    """Logarithmic function: a * log(b * x + c)"""
    return a * np.log(b * x + c)

# Calculate the rate of change (first differences) for raw data
def calculate_rate_of_change(x, y):
    """Calculate rate of change between adjacent points."""
    roc = np.zeros(len(x) - 1)
    x_mid = np.zeros(len(x) - 1)
    
    for i in range(len(y) - 1):
        # Calculate difference in y divided by difference in x
        roc[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
        # Use midpoint between x values as the x-coordinate for the rate of change
        x_mid[i] = (x[i+1] + x[i]) / 2
    
    return x_mid, roc

# Function to fit models and return results
def fit_models(x, y, title):
    models = {
        'Logistic': (logistic, [0.95, 0.05, 20, 0.75]),  # Adjusted initial params for the new data
        'Exponential Approach': (exp_approach, [0.2, 0.05, 0.75]),
        'Logarithmic': (log_func, [0.05, 0.5, 1.0])
    }

    results = {}
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Data')
    
    for name, (func, p0) in models.items():
        try:
            # Fit the model
            params, covariance = curve_fit(func, x, y, p0=p0, maxfev=10000)
            
            # Generate predictions for plotting
            x_smooth = np.linspace(min(x), max(x), 1000)
            y_pred_smooth = func(x_smooth, *params)
            
            # Generate predictions for the original x values for error calculation
            y_pred = func(x, *params)
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            # Calculate R-squared
            ss_tot = np.sum((y - np.mean(y))**2)
            ss_res = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Plot the fitted curve
            plt.plot(x_smooth, y_pred_smooth, label=f'{name} (RMSE: {rmse:.4f})')
            
            # Create equation string
            if name == 'Logistic':
                L, k, x0, b = params
                equation = f"y = {L:.4f} / (1 + exp(-{k:.4f} * (x - {x0:.4f}))) + {b:.4f}"
            elif name == 'Exponential Approach':
                a, b, c = params
                equation = f"y = {a:.4f} * (1 - exp(-{b:.4f} * x)) + {c:.4f}"
            elif name == 'Logarithmic':
                a, b, c = params
                equation = f"y = {a:.4f} * log({b:.4f} * x + {c:.4f})"
            
            # Store results
            results[name] = {
                'params': params,
                'rmse': rmse,
                'r_squared': r_squared,
                'equation': equation,
                'function': func,
                'y_pred_smooth': y_pred_smooth,
                'x_smooth': x_smooth
            }
            
            print(f"Successfully fit {name} model for {title}")
            print(f"  RMSE: {rmse:.6f}, R-squared: {r_squared:.6f}")
            print(f"  Equation: {equation}")
            
        except Exception as e:
            print(f"Failed to fit {name} for {title}: {e}")
    
    plt.title(f'{title} - All Curve Fits')
    plt.xlabel('Dimension')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace("@", "_at_")}_all_fits.png')
    
    return results

# Calculate the analytical derivative for each fitted function
def calculate_analytical_derivatives(results):
    """Calculate analytical derivatives for fitted functions."""
    derivatives = {}
    
    for name, model_data in results.items():
        func = model_data['function']
        params = model_data['params']
        x_smooth = model_data['x_smooth']
        
        # Calculate analytical derivative based on function type
        if name == 'Logistic':
            L, k, x0, b = params
            # Derivative of logistic: L*k*exp(-k*(x-x0)) / (1 + exp(-k*(x-x0)))^2
            numerator = L * k * np.exp(-k * (x_smooth - x0))
            denominator = (1 + np.exp(-k * (x_smooth - x0)))**2
            dy_dx = numerator / denominator
        
        elif name == 'Exponential Approach':
            a, b, c = params
            # Derivative of a*(1-exp(-b*x))+c is a*b*exp(-b*x)
            dy_dx = a * b * np.exp(-b * x_smooth)
        
        elif name == 'Logarithmic':
            a, b, c = params
            # Derivative of a*log(b*x+c) is a*b/(b*x+c)
            dy_dx = a * b / (b * x_smooth + c)
        
        derivatives[name] = (x_smooth, dy_dx)
    
    return derivatives

# Fit all models to precision@10
print("Fitting models to Precision@10...")
results10 = fit_models(x, y10, 'Precision@10')

# Print detailed results for all models
print("\n======= DETAILED RESULTS =======")
print(f"\n----- Precision@10 -----")

if not results10:
    print("No successful model fits")
else:
    # Sort models by RMSE (lower is better)
    sorted_models = sorted(results10.items(), key=lambda x: x[1]['rmse'])
    
    for i, (model_name, model_data) in enumerate(sorted_models):
        print(f"\n{i+1}. {model_name}")
        print(f"   Equation: {model_data['equation']}")
        print(f"   RMSE: {model_data['rmse']:.6f}")
        print(f"   R-squared: {model_data['r_squared']:.6f}")
        print(f"   Parameters: {model_data['params']}")

# Calculate rate of change for the original data
x_mid, roc = calculate_rate_of_change(x, y10)

# Make a scatterplot of the rate of change vs. dimension
plt.figure(figsize=(12, 7))
plt.scatter(x_mid, roc, color='blue', s=80, alpha=0.7, label='Data points ROC')

# Add line connecting the points to better visualize the trend
plt.plot(x_mid, roc, 'b-', alpha=0.5)

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Calculate and plot the analytical derivatives for the best fit model
if results10:
    best_model = min(results10.items(), key=lambda x: x[1]['rmse'])
    model_name = best_model[0]
    model_data = best_model[1]
    
    # Calculate derivatives for all models
    derivatives = calculate_analytical_derivatives(results10)
    
    # Get derivative of the best model
    x_smooth, dy_dx_best = derivatives[model_name]
    
    # Plot the analytical derivative of the best model
    plt.plot(x_smooth, dy_dx_best, 'r-', linewidth=2, 
             label=f'Best model ({model_name}) derivative')
    
    # Add horizontal lines at specific rate of change thresholds for reference
    threshold_values = [0.001, 0.002, 0.003, 0.004, 0.005]
    for threshold in threshold_values:
        plt.axhline(y=threshold, color='green', linestyle=':', alpha=0.5)
        # Find dimension where rate of change crosses the threshold
        for i in range(len(x_smooth)-1):
            if (dy_dx_best[i] >= threshold and dy_dx_best[i+1] < threshold) or \
               (dy_dx_best[i] <= threshold and dy_dx_best[i+1] > threshold):
                intercept_x = x_smooth[i] + (threshold - dy_dx_best[i]) * \
                             (x_smooth[i+1] - x_smooth[i]) / (dy_dx_best[i+1] - dy_dx_best[i])
                plt.text(intercept_x + 2, threshold + 0.0002, 
                         f'dim={intercept_x:.1f}', fontsize=9)
                plt.scatter(intercept_x, threshold, color='green', s=30, zorder=5)

plt.title('Rate of Change in Precision@10 vs Dimension')
plt.xlabel('Dimension')
plt.ylabel('Rate of Change (Δprecision/Δdimension)')
plt.grid(True)
plt.legend()

# Create a table to display numeric values of ROC for each dimension
table_text = []
table_text.append('Dimension Midpoint | Rate of Change')
table_text.append('-' * 40)
for i in range(len(x_mid)):
    table_text.append(f"{x_mid[i]:10.1f} | {roc[i]:.6f}")

plt.figtext(0.02, 0.02, '\n'.join(table_text), fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('precision10_rate_of_change.png')

# Plot showing smooth rate of change for all fitted models
plt.figure(figsize=(12, 7))

# Plot the rate of change data points
plt.scatter(x_mid, roc, color='blue', s=60, alpha=0.6, label='Data points ROC')
plt.plot(x_mid, roc, 'b-', alpha=0.4)

# Calculate and plot derivatives for all models
derivatives = calculate_analytical_derivatives(results10)
colors = ['red', 'green', 'purple']

for i, (name, (x_deriv, y_deriv)) in enumerate(derivatives.items()):
    plt.plot(x_deriv, y_deriv, color=colors[i % len(colors)], linewidth=2, 
             label=f'{name} derivative')

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.title('Rate of Change Comparison for All Models')
plt.xlabel('Dimension')
plt.ylabel('Rate of Change (Δprecision/Δdimension)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('precision10_all_derivatives.png')

# Print the rate of change values
print("\n======= RATE OF CHANGE ANALYSIS =======")
print("Dimension Midpoint | Rate of Change")
print("-" * 40)
for i in range(len(x_mid)):
    print(f"{x_mid[i]:10.1f} | {roc[i]:.6f}")

# Create a DataFrame to sort the rate of change values
roc_df = pd.DataFrame({
    'Dimension Midpoint': x_mid,
    'Rate of Change': roc
})

# Sort the DataFrame by rate of change in descending order
sorted_roc = roc_df.sort_values('Rate of Change', ascending=False)

# Print the sorted rate of change values
print("\n======= RATE OF CHANGE SORTED (LARGEST TO SMALLEST) =======")
print("Dimension Midpoint | Rate of Change")
print("-" * 40)
for idx, row in sorted_roc.iterrows():
    print(f"{row['Dimension Midpoint']:10.1f} | {row['Rate of Change']:.6f}")

# Create a bar chart of sorted rate of change values
plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(sorted_roc)), sorted_roc['Rate of Change'], color='skyblue')

# Add dimension labels to the bars
plt.xticks(range(len(sorted_roc)), [f"{dim:.0f}" for dim in sorted_roc['Dimension Midpoint']], rotation=45)

# Add value labels above each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f"{sorted_roc['Rate of Change'].iloc[i]:.4f}",
            ha='center', va='bottom', rotation=0, fontsize=9)

plt.title('Rate of Change in Precision@10 (Sorted from Largest to Smallest)')
plt.xlabel('Dimension Midpoint')
plt.ylabel('Rate of Change (Δprecision/Δdimension)')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('precision10_sorted_rate_of_change.png')

# If we have a best model, find where the derivative drops below thresholds
if results10:
    best_model = min(results10.items(), key=lambda x: x[1]['rmse'])
    model_name = best_model[0]
    
    print(f"\nRate of Change Thresholds for Best Model ({model_name}):")
    print("-" * 60)
    
    x_smooth, dy_dx_best = derivatives[model_name]
    thresholds = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for threshold in thresholds:
        # Find where derivative drops below threshold
        for i in range(len(x_smooth)-1):
            if dy_dx_best[i] >= threshold and dy_dx_best[i+1] < threshold:
                intercept_x = x_smooth[i] + (threshold - dy_dx_best[i]) * \
                            (x_smooth[i+1] - x_smooth[i]) / (dy_dx_best[i+1] - dy_dx_best[i])
                print(f"Rate of change = {threshold:.4f} at dimension ≈ {intercept_x:.1f}")
                break
        else:
            # If we never cross this threshold
            if dy_dx_best[-1] > threshold:
                print(f"Rate of change never drops below {threshold:.4f}")
            else:
                print(f"Rate of change is already below {threshold:.4f} at dimension = {x_smooth[0]:.1f}")
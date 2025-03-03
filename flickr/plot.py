import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = pd.read_csv('data/precision.csv')
x = data['reduced_dimension'].values
y5 = data['precision_at_5'].values
y10 = data['precision_at_10'].values
y15 = data['precision_at_15'].values

# Define candidate functions to fit
def logistic(x, L, k, x0, b):
    """Logistic function: L / (1 + exp(-k * (x - x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b

def gompertz(x, a, b, c):
    """Gompertz function: a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))

def hill(x, a, b, c, d):
    """Hill equation: a + (b - a) * (x^c / (d^c + x^c))"""
    return a + (b - a) * (x**c / (d**c + x**c))

def exp_approach(x, a, b, c):
    """Exponential approach: a * (1 - exp(-b * x)) + c"""
    return a * (1 - np.exp(-b * x)) + c

# Define a logarithmic function
def log_func(x, a, b, c):
    """Logarithmic function: a * log(b * x + c)"""
    return a * np.log(b * x + c)

# Function to fit models and return results
def fit_models(x, y, title):
    models = {
        'Logistic': (logistic, [0.3, 0.02, 100, 0.7]),
        'Gompertz': (gompertz, [1.0, 5.0, 0.01]),
        'Hill': (hill, [0.7, 1.0, 1.0, 100]),
        'Exponential Approach': (exp_approach, [0.3, 0.01, 0.7]),
        'Logarithmic': (log_func, [0.1, 0.01, 1.0])
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
            elif name == 'Gompertz':
                a, b, c = params
                equation = f"y = {a:.4f} * exp(-{b:.4f} * exp(-{c:.4f} * x))"
            elif name == 'Hill':
                a, b, c, d = params
                equation = f"y = {a:.4f} + ({b:.4f} - {a:.4f}) * (x^{c:.4f} / ({d:.4f}^{c:.4f} + x^{c:.4f}))"
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
                'function': func
            }
            
            print(f"Successfully fit {name} model for {title}")
            print(f"  RMSE: {rmse:.6f}, R-squared: {r_squared:.6f}")
            print(f"  Equation: {equation}")
            
        except Exception as e:
            print(f"Failed to fit {name} for {title}: {e}")
    
    plt.title(f'{title} - All Curve Fits')
    plt.xlabel('Reduced Dimension')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace("@", "_at_")}_all_fits.png')
    
    return results

# Fit all models to each precision curve
print("Fitting models to Precision@5...")
results5 = fit_models(x, y5, 'Precision@5')

print("\nFitting models to Precision@10...")
results10 = fit_models(x, y10, 'Precision@10')

print("\nFitting models to Precision@15...")
results15 = fit_models(x, y15, 'Precision@15')

# Print detailed results for all models and metrics
print("\n======= DETAILED RESULTS =======")

for metric, results, ydata in [("Precision@5", results5, y5), 
                              ("Precision@10", results10, y10), 
                              ("Precision@15", results15, y15)]:
    
    print(f"\n----- {metric} -----")
    
    if not results:
        print("No successful model fits")
        continue
    
    # Sort models by RMSE (lower is better)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
    
    for i, (model_name, model_data) in enumerate(sorted_models):
        print(f"\n{i+1}. {model_name}")
        print(f"   Equation: {model_data['equation']}")
        print(f"   RMSE: {model_data['rmse']:.6f}")
        print(f"   R-squared: {model_data['r_squared']:.6f}")
        print(f"   Parameters: {model_data['params']}")

# Create a visual comparison of the best model for each metric
plt.figure(figsize=(12, 8))

# For each metric, plot data and its best fit (lowest RMSE)
colors = ['blue', 'green', 'red']
for i, (metric, results, ydata, color) in enumerate([
    ("Precision@5", results5, y5, colors[0]),
    ("Precision@10", results10, y10, colors[1]),
    ("Precision@15", results15, y15, colors[2])
]):
    
    if not results:
        print(f"No successful fits for {metric}, skipping plot")
        continue
    
    # Get the best model (lowest RMSE)
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    model_name = best_model[0]
    model_data = best_model[1]
    
    # Plot data points
    plt.scatter(x, ydata, color=color, label=f'{metric} (Data)')
    
    # Plot best fit curve
    x_smooth = np.linspace(min(x), max(x), 1000)
    func = model_data['function']
    y_pred = func(x_smooth, *model_data['params'])
    
    plt.plot(x_smooth, y_pred, color=color, linestyle='-', 
             label=f'{metric} Best: {model_name}')

plt.title('Best Fitted Models for Each Precision Metric')
plt.xlabel('Reduced Dimension')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('precision_curves_best_models.png')

# Print summary of best models
print("\n======= SUMMARY OF BEST MODELS =======")
for metric, results in [("Precision@5", results5), 
                      ("Precision@10", results10), 
                      ("Precision@15", results15)]:
    
    if not results:
        print(f"No successful models for {metric}")
        continue
    
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    model_name = best_model[0]
    model_data = best_model[1]
    
    print(f"\n{metric}: {model_name}")
    print(f"Equation: {model_data['equation']}")
    print(f"RMSE: {model_data['rmse']:.6f}")
    print(f"R-squared: {model_data['r_squared']:.6f}")
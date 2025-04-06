import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from precision4.csv
data = pd.read_csv('data/precision.csv')
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
                'equation': equation
            }
            
            print(f"Successfully fit {name} model for {title}")
            print(f"  RMSE: {rmse:.6f}, R-squared: {r_squared:.6f}")
            print(f"  Equation: {equation}")
            
        except Exception as e:
            print(f"Failed to fit {name} for {title}: {e}")
    
    plt.title(f'{title} - Curve Fits')
    plt.xlabel('Dimension')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace("@", "_at_")}_fits.png')
    
    return results

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

plt.show()  # Display the plot
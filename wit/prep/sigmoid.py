import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error

# Define fitting models
# ---------------------
# Logarithmic model
def log_model(x, a, b):
    # Add small constant to handle x=0
    return a * np.log(x + 1) + b

# Original Sigmoid model (logistic function)
def sigmoid_model(x, a, b, c, d):
    """
    Parameters:
    a: scale factor
    b: steepness factor
    c: midpoint (inflection point on x-axis)
    d: vertical offset
    """
    return a / (1 + np.exp(-b * (x - c))) + d

# Function to perform grid search for initial parameters
def grid_search_sigmoid_params(x, y):
    """
    Perform a grid search to find good initial parameters for sigmoid fitting
    """
    # Define parameter ranges to search
    a_values = np.linspace(800, 1200, 5)  # Scale factor range
    b_values = np.linspace(0.005, 0.05, 5)  # Steepness range
    c_values = np.linspace(-700, -400, 5)  # Midpoint range
    d_values = np.linspace(-1200, -800, 5)  # Offset range
    
    best_error = float('inf')
    best_params = None
    
    # Simple grid search
    for a in a_values:
        for b in b_values:
            for c in c_values:
                for d in d_values:
                    y_pred = sigmoid_model(x, a, b, c, d)
                    error = np.sum((y - y_pred) ** 2)  # Sum of squared errors
                    
                    if error < best_error:
                        best_error = error
                        best_params = (a, b, c, d)
    
    return best_params

# Function to find best sigmoid fit with multiple optimization approaches
def find_best_sigmoid_fit(x, y):
    """
    Try multiple optimization approaches to find the best sigmoid fit
    """
    x_array = np.array(x)
    y_array = np.array(y)
    
    # Calculate key statistics for setting bounds
    y_min = np.min(y_array)
    y_max = np.max(y_array)
    y_range = y_max - y_min
    x_median = np.median(x_array)
    
    # For the original model, we'll allow larger parameter values
    # But still with some reasonable constraints
    bounds = (
        [0.1, 0.001, -1000, -1200],  # Lower bounds
        [1500, 0.1, 1000, 0.1]  # Upper bounds
    )
    
    # Get initial params from grid search
    initial_params = grid_search_sigmoid_params(x_array, y_array)
    print(f"Grid search initial params: {initial_params}")
    
    # Try different optimization methods
    methods = ['trf', 'dogbox']
    best_fit = None
    best_rmse = float('inf')
    
    for method in methods:
        try:
            params, cov = curve_fit(
                sigmoid_model, 
                x_array, 
                y_array, 
                p0=initial_params,
                bounds=bounds,
                method=method,
                maxfev=10000
            )
            
            # Calculate predictions and metrics
            y_pred = sigmoid_model(x_array, *params)
            rmse = np.sqrt(mean_squared_error(y_array, y_pred))
            r2 = r2_score(y_array, y_pred)
            
            print(f"Method: {method}, RMSE: {rmse:.6f}, R²: {r2:.6f}")
            print(f"Parameters: {params}")
            
            # Keep track of best fit
            if rmse < best_rmse:
                best_rmse = rmse
                best_fit = {
                    'params': params,
                    'method': method,
                    'rmse': rmse,
                    'r2': r2,
                    'predictions': y_pred,
                    'function': lambda x, p=params: sigmoid_model(x, *p)
                }
        except Exception as e:
            print(f"Error with method {method}: {e}")
    
    # Try differential evolution as an alternative approach
    try:
        def objective(params):
            a, b, c, d = params
            y_pred = sigmoid_model(x_array, a, b, c, d)
            return np.sum((y_array - y_pred) ** 2)
        
        result = differential_evolution(
            objective, 
            bounds=[
                (0.1, 1500),          # a
                (0.001, 0.1),         # b
                (-1000, 1000),        # c
                (-1200, 0.1)          # d
            ],
            maxiter=1000,
            popsize=15
        )
        
        params = result.x
        y_pred = sigmoid_model(x_array, *params)
        rmse = np.sqrt(mean_squared_error(y_array, y_pred))
        r2 = r2_score(y_array, y_pred)
        
        print(f"Method: differential_evolution, RMSE: {rmse:.6f}, R²: {r2:.6f}")
        print(f"Parameters: {params}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_fit = {
                'params': params,
                'method': 'differential_evolution',
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'function': lambda x, p=params: sigmoid_model(x, *p)
            }
    except Exception as e:
        print(f"Error with differential evolution: {e}")
    
    return best_fit

# 1. Data Exploration and Preparation
# ----------------------------------
# Load the data
data = pd.read_csv('data/precision.csv')

# Print basic information
print("Data shape:", data.shape)

# Calculate summary statistics
stats = data.describe()
print("\nData summary:")
print(stats)

# 2. Visualize Raw Data
# ---------------------
plt.figure(figsize=(12, 8))
plt.plot(data['dimension'], data['precision_at_5'], 'o', label='k=5')
plt.plot(data['dimension'], data['precision_at_10'], 's', label='k=10')
plt.plot(data['dimension'], data['precision_at_15'], '^', label='k=15')
plt.xlabel('Dimension')
plt.ylabel('Precision')
plt.title('Raw Precision Data at Different k Values')
plt.legend()
plt.grid(True)
plt.savefig('raw_data_plot.png')
plt.close()

# 3. Fit Models and Evaluate
# -------------------------
k_values = ['precision_at_5', 'precision_at_10', 'precision_at_15']
k_labels = ['k=5', 'k=10', 'k=15']
all_results = {}
best_models = {}

# Fit logarithmic model
for k_value, k_label in zip(k_values, k_labels):
    print(f"\nFitting logarithmic model for {k_label}...")
    x_array = np.array(data['dimension'])
    y_array = np.array(data[k_value])
    
    try:
        log_params, log_cov = curve_fit(log_model, x_array, y_array)
        log_pred = log_model(x_array, *log_params)
        log_r2 = r2_score(y_array, log_pred)
        log_rmse = np.sqrt(mean_squared_error(y_array, log_pred))
        
        print(f"Logarithmic model: R² = {log_r2:.4f}, RMSE = {log_rmse:.4f}")
        print(f"Parameters: {log_params}")
        
        if k_label not in all_results:
            all_results[k_label] = {}
        
        all_results[k_label]['Logarithmic'] = {
            'params': log_params,
            'r2': log_r2,
            'rmse': log_rmse,
            'predictions': log_pred,
            'function': lambda x, p=log_params: log_model(x, *p)
        }
    except Exception as e:
        print(f"Error fitting logarithmic model for {k_label}: {e}")

# Fit improved sigmoid model
for k_value, k_label in zip(k_values, k_labels):
    print(f"\nFitting improved sigmoid model for {k_label}...")
    
    best_sigmoid = find_best_sigmoid_fit(data['dimension'], data[k_value])
    
    if best_sigmoid:
        print(f"Best sigmoid fit for {k_label}:")
        print(f"  Method: {best_sigmoid['method']}")
        print(f"  R²: {best_sigmoid['r2']:.4f}")
        print(f"  RMSE: {best_sigmoid['rmse']:.4f}")
        print(f"  Parameters: {best_sigmoid['params']}")
        
        if k_label not in all_results:
            all_results[k_label] = {}
        
        all_results[k_label]['Sigmoid'] = best_sigmoid
    
    # Determine best model for this k value
    if k_label in all_results:
        best_model = max(all_results[k_label].items(), key=lambda x: x[1]['r2'])
        best_models[k_label] = best_model
        print(f"Best model for {k_label}: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

# 4. Visualize Fitted Models
# -------------------------
# Create smooth x values for plotting
x_smooth = np.linspace(0, max(data['dimension']), 1000)

# Plot fitted models for each k value
for i, (k_value, k_label) in enumerate(zip(k_values, k_labels)):
    plt.figure(figsize=(12, 8))
    
    # Plot raw data
    plt.scatter(data['dimension'], data[k_value], label='Raw data', color='black', marker='o')
    
    # Plot mean line
    mean_val = data[k_value].mean()
    plt.axhline(y=mean_val, color='orange', linestyle='--', 
                label=f'Mean = {mean_val:.4f}')
    
    # Plot fitted models
    colors = ['blue', 'green']
    for j, (model_name, model_results) in enumerate(all_results[k_label].items()):
        # Generate predictions for smooth curve
        y_smooth = np.array([model_results['function'](x) for x in x_smooth])
        
        plt.plot(x_smooth, y_smooth, label=f"{model_name} (R² = {model_results['r2']:.4f})", 
                color=colors[j % len(colors)], linewidth=2)
    
    plt.xlabel('Dimension')
    plt.ylabel(f'Precision at {k_label.split("=")[1]}')
    plt.title(f'Model Fitting for Precision at {k_label.split("=")[1]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'improved_model_fitting_{k_label.replace("=", "")}.png')
    plt.close()

# 5. Plot Residuals for Best Models
# --------------------------------
for k_value, k_label in zip(k_values, k_labels):
    if k_label in best_models:
        best_model_name, best_model_results = best_models[k_label]
        
        plt.figure(figsize=(12, 6))
        
        # Calculate residuals
        residuals = data[k_value] - best_model_results['predictions']
        
        # Plot residuals
        plt.scatter(data['dimension'], residuals, color='blue', marker='o')
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xlabel('Dimension')
        plt.ylabel('Residuals')
        plt.title(f'Residuals for Improved {best_model_name} Model - {k_label}')
        plt.grid(True)
        plt.savefig(f'improved_residuals_{k_label.replace("=", "")}.png')
        plt.close()
        
        # Additional: Plot residual histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, color='blue', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='-')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title(f'Residual Distribution for {best_model_name} Model - {k_label}')
        plt.grid(True)
        plt.savefig(f'improved_residual_hist_{k_label.replace("=", "")}.png')
        plt.close()

# 6. Print Summary Report
# ----------------------
print("\n=== IMPROVED MODEL SUMMARY REPORT ===")
print("\nBest Models:")
for k_label, (model_name, model_info) in best_models.items():
    print(f"\n{k_label}:")
    print(f"  Best model: {model_name}")
    print(f"  R²: {model_info['r2']:.4f}")
    print(f"  RMSE: {model_info['rmse']:.4f}")
    
    if model_name == 'Logarithmic':
        a, b = model_info['params']
        print(f"  Equation: y = {a:.6f}*log(x+1) + {b:.6f}")
    elif model_name == 'Sigmoid':
        a, b, c, d = model_info['params']
        print(f"  Equation: y = {a:.6f}/(1+exp(-{b:.6f}*(x-{c:.6f}))) + {d:.6f}")
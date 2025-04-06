import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# Define fitting models
# ---------------------
# Logarithmic model
def log_model(x, a, b):
    # Add small constant to handle x=0
    return a * np.log(x + 1) + b

# Sigmoid model (logistic function)
def sigmoid_model(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

# 1. Data Exploration and Preparation
# ----------------------------------
# Load the data
data = pd.read_csv('data/precision.csv')

# Print basic information
print("Data shape:", data.shape)
print("\nFirst few rows:")
print(data.head())

# Calculate summary statistics
stats = data.describe()

# Calculate midpoints and add to stats dataframe
for col in ['precision_at_5', 'precision_at_10', 'precision_at_15']:
    min_val = stats.loc['min', col]
    max_val = stats.loc['max', col]
    midpoint = (min_val + max_val) / 2
    stats.loc['midpoint', col] = midpoint

print("\nData summary:")
print(stats)

# 3. Fit Models and Evaluate
# -------------------------
# Function to fit and evaluate models
def fit_and_evaluate_models(x, y, k_value):
    x_array = np.array(x)
    y_array = np.array(y)
    
    results = {}
    
    # Logarithmic fit
    try:
        log_params, log_cov = curve_fit(log_model, x_array, y_array)
        log_pred = log_model(x_array, *log_params)
        log_r2 = r2_score(y_array, log_pred)
        log_rmse = np.sqrt(mean_squared_error(y_array, log_pred))
        results['Logarithmic'] = {
            'params': log_params,
            'r2': log_r2,
            'rmse': log_rmse,
            'predictions': log_pred,
            'function': lambda x: log_model(x, *log_params)
        }
    except Exception as e:
        print(f"Error fitting logarithmic model for k={k_value}: {e}")
    
    # Sigmoid fit (logistic function)
    try:
        # Initial guesses based on data
        p0 = [max(y_array) - min(y_array), 0.01, np.median(x_array), min(y_array)]
        sigmoid_params, sigmoid_cov = curve_fit(sigmoid_model, x_array, y_array, p0=p0, maxfev=10000)
        sigmoid_pred = sigmoid_model(x_array, *sigmoid_params)
        sigmoid_r2 = r2_score(y_array, sigmoid_pred)
        sigmoid_rmse = np.sqrt(mean_squared_error(y_array, sigmoid_pred))
        results['Sigmoid'] = {
            'params': sigmoid_params,
            'r2': sigmoid_r2,
            'rmse': sigmoid_rmse,
            'predictions': sigmoid_pred,
            'function': lambda x: sigmoid_model(x, *sigmoid_params)
        }
    except Exception as e:
        print(f"Error fitting sigmoid model for k={k_value}: {e}")
    
    # Find best model based on R²
    best_model = max(results.items(), key=lambda x: x[1]['r2']) if results else None
    
    return results, best_model

k_values = ['precision_at_5', 'precision_at_10', 'precision_at_15']
k_labels = ['k=5', 'k=10', 'k=15']
all_results = {}
best_models = {}

for k_value, k_label in zip(k_values, k_labels):
    print(f"\nFitting models for {k_label}...")
    results, best_model = fit_and_evaluate_models(data['dimension'], data[k_value], k_label)
    all_results[k_label] = results
    best_models[k_label] = best_model
    
    # Print evaluation metrics
    print(f"Results for {k_label}:")
    for model_name, model_results in results.items():
        print(f"  {model_name}: R² = {model_results['r2']:.4f}, RMSE = {model_results['rmse']:.4f}")
    
    if best_model:
        print(f"Best model for {k_label}: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")

# 8. Print Summary Report
# ----------------------
print("\n=== SUMMARY REPORT ===")
print("\nBest Models:")
for k_label, (model_name, model_info) in best_models.items():
    print(f"\n{k_label}:")
    print(f"  Best model: {model_name}")
    print(f"  R²: {model_info['r2']:.6f}")
    print(f"  RMSE: {model_info['rmse']:.6f}")
    
    if model_name == 'Logarithmic':
        a, b = model_info['params']
        print(f"  Equation: y = {a:.6f}*log(x+1) + {b:.6f}")
    elif model_name == 'Sigmoid':
        a, b, c, d = model_info['params']
        print(f"  Equation: y = {a:.6f}/(1+exp(-{b:.6f}*(x-{c:.6f}))) + {d:.6f}")
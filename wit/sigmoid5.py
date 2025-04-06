import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_squared_error

# Set up plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (18, 5)  # Wider figure for horizontal alignment
plt.rcParams['font.size'] = 12

# Define sigmoid model (logistic function)
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
def find_best_sigmoid_fit(x, y, seed=None):
    """
    Try multiple optimization approaches to find the best sigmoid fit
    """
    x_array = np.array(x)
    y_array = np.array(y)
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate key statistics for setting bounds
    y_min = np.min(y_array)
    y_max = np.max(y_array)
    y_range = y_max - y_min
    
    # For the original model with reasonable constraints
    bounds = (
        [0.1, 0.001, -1000, -1200],  # Lower bounds
        [1500, 0.1, 1000, 0.1]       # Upper bounds
    )
    
    # Get initial params from grid search
    initial_params = grid_search_sigmoid_params(x_array, y_array)
    
    # Try different optimization methods
    methods = ['trf', 'dogbox']
    best_fit = None
    best_rmse = float('inf')
    
    for method in methods:
        try:
            params, _ = curve_fit(
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
        except Exception:
            pass
    
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
            popsize=15,
            seed=seed
        )
        
        params = result.x
        y_pred = sigmoid_model(x_array, *params)
        rmse = np.sqrt(mean_squared_error(y_array, y_pred))
        r2 = r2_score(y_array, y_pred)
        
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
    except Exception:
        pass
    
    return best_fit

# Function to create individual high-quality sigmoid plots
def create_high_quality_sigmoid_plot(data, k_value, k_label, sigmoid_result, save_path=None):
    """
    Create a high-quality plot showing sigmoid fit for a specific k value
    """
    # Create figure with high DPI for quality
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Generate smooth x values for plotting curves
    x_smooth = np.linspace(min(data['dimension']), max(data['dimension']), 1000)
    
    # Plot data points
    ax.scatter(data['dimension'], data[k_value], color='blue', alpha=0.7, s=50, label='Data')
    
    # Plot sigmoid fit
    if sigmoid_result is not None:
        params = sigmoid_result['params']
        y_sigmoid = sigmoid_model(x_smooth, *params)
        r2 = sigmoid_result['r2']
        ax.plot(x_smooth, y_sigmoid, 'r-', linewidth=3, label=f'Sigmoid (R²={r2:.4f})')
    
    # Add labels and title with larger font sizes
    ax.set_xlabel('Dimension', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision at {k_label.split("=")[1]}', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set consistent y-axis limits for better comparison between plots
    ax.set_ylim(0.6, 1.05)
    
    # Add equation text in a visible box
    if sigmoid_result is not None:
        a, b, c, d = sigmoid_result['params']
        equation = f'y = {a:.2f}/(1+exp(-{b:.5f}*(x-{c:.2f}))) + {d:.2f}'
        ax.text(0.5, 0.03, equation, transform=ax.transAxes, ha='center', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add statistical metrics
    if sigmoid_result is not None:
        stats_text = f"R² = {sigmoid_result['r2']:.6f}\nRMSE = {sigmoid_result['rmse']:.6f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add grid lines for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save with high resolution if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Main execution
def main():
    # Load the data
    data = pd.read_csv('data/precision.csv')

    # Run multiple trials
    num_trials = 100
    k_values = ['precision_at_5', 'precision_at_10', 'precision_at_15']
    k_labels = ['k=5', 'k=10', 'k=15']

    # Storage for best results only
    best_overall_results = {k: {'r2': 0, 'result': None} for k in k_labels}

    print(f"Running {num_trials} trials for each k value...")

    # Run sigmoid model fitting multiple times
    for trial in range(num_trials):
        trial_seed = trial * 42  # Different seed for each trial
        
        for k_value, k_label in zip(k_values, k_labels):
            # Fit sigmoid model with current seed
            best_sigmoid = find_best_sigmoid_fit(data['dimension'], data[k_value], seed=trial_seed)
            
            if best_sigmoid:
                # Update best overall result if better
                if best_sigmoid['r2'] > best_overall_results[k_label]['r2']:
                    best_overall_results[k_label] = {
                        'r2': best_sigmoid['r2'],
                        'result': {
                            'trial': trial + 1,
                            'method': best_sigmoid['method'],
                            'params': best_sigmoid['params'],
                            'r2': best_sigmoid['r2'],
                            'rmse': best_sigmoid['rmse']
                        }
                    }

    # Print summary report
    print("\n=== SUMMARY REPORT ===")

    # Print only the best result for each k
    print("\nBest Sigmoid Results (from 100 trials):")
    for k_label in k_labels:
        if best_overall_results[k_label]['result']:
            best_result = best_overall_results[k_label]['result']
            a, b, c, d = best_result['params']
            print(f"\n{k_label}:")
            print(f"  Best from Trial: {best_result['trial']}")
            print(f"  Method: {best_result['method']}")
            print(f"  R²: {best_result['r2']:.10f}")
            print(f"  RMSE: {best_result['rmse']:.10f}")
            print(f"  Equation: y = {a:.6f}/(1+exp(-{b:.6f}*(x-{c:.6f}))) + {d:.6f}")

    # Create individual high-quality plots
    print("\nCreating individual high-quality sigmoid plots...")
    for k_value, k_label in zip(k_values, k_labels):
        if best_overall_results[k_label]['result']:
            # Get best sigmoid result for this k value
            sigmoid_result = {
                'params': best_overall_results[k_label]['result']['params'],
                'r2': best_overall_results[k_label]['result']['r2'],
                'rmse': best_overall_results[k_label]['result']['rmse']
            }
            
            # Create and save high-quality plot
            fig = create_high_quality_sigmoid_plot(
                data, 
                k_value, 
                k_label, 
                sigmoid_result,
                save_path=f"sigmoid_fit_{k_label.replace('=', '')}_high_quality.png"
            )
            plt.close(fig)  # Close to prevent display in non-interactive environments

    print("\nVisualization completed! Saved three high-quality plots to working directory.")

if __name__ == "__main__":
    main()